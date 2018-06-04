// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
// vi: set et ts=4 sw=4 sts=4:
/*
  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.

  Consult the COPYING file in the top-level source directory of this
  module for the precise wording of the license and the list of
  copyright holders.
*/
/*!
 * \file
 * \copydoc Ewoms::Linear::FemSolverBackend
 */
#ifndef EWOMS_FEM_SOLVER_BACKEND_HH
#define EWOMS_FEM_SOLVER_BACKEND_HH

#if HAVE_DUNE_FEM

#include <dune/fem/solver/istlsolver.hh>

#include <ewoms/common/genericguard.hh>
#include <ewoms/common/propertysystem.hh>
#include <ewoms/common/parametersystem.hh>

#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/common/fvector.hh>

#include <sstream>
#include <memory>
#include <iostream>

namespace Ewoms {
namespace Properties {
NEW_TYPE_TAG(ParallelBaseLinearSolver);

// forward declaration of the required property tags
NEW_PROP_TAG(Simulator);
NEW_PROP_TAG(Scalar);
NEW_PROP_TAG(NumEq);
NEW_PROP_TAG(JacobianMatrix);
NEW_PROP_TAG(GlobalEqVector);
NEW_PROP_TAG(VertexMapper);
NEW_PROP_TAG(GridView);

NEW_PROP_TAG(BorderListCreator);
NEW_PROP_TAG(Overlap);
NEW_PROP_TAG(OverlappingVector);
NEW_PROP_TAG(OverlappingMatrix);
NEW_PROP_TAG(OverlappingScalarProduct);
NEW_PROP_TAG(OverlappingLinearOperator);

//! The type of the linear solver to be used
NEW_PROP_TAG(LinearSolverBackend);

//! the preconditioner used by the linear solver
NEW_PROP_TAG(PreconditionerWrapper);


//! The floating point type used internally by the linear solver
NEW_PROP_TAG(LinearSolverScalar);

/*!
 * \brief Maximum accepted error of the solution of the linear solver.
 */
NEW_PROP_TAG(LinearSolverTolerance);

/*!
 * \brief Specifies the verbosity of the linear solver
 *
 * By default it is 0, i.e. it doesn't print anything. Setting this
 * property to 1 prints aggregated convergence rates, 2 prints the
 * convergence rate of every iteration of the scheme.
 */
NEW_PROP_TAG(LinearSolverVerbosity);

//! Maximum number of iterations eyecuted by the linear solver
NEW_PROP_TAG(LinearSolverMaxIterations);

//! The order of the sequential preconditioner
NEW_PROP_TAG(PreconditionerOrder);

//! The relaxation factor of the preconditioner
NEW_PROP_TAG(PreconditionerRelaxation);
}} // namespace Properties, Ewoms

namespace Ewoms {
namespace Linear {
/*!
 * \ingroup Linear
 *
 * \brief Provides the common code which is required by most linear solvers.
 *
 * This class provides access to all preconditioners offered by dune-istl using the
 * PreconditionerWrapper property:
 * \code
 * SET_TYPE_PROP(YourTypeTag, PreconditionerWrapper,
 *               Ewoms::Linear::PreconditionerWrapper$PRECONDITIONER<TypeTag>);
 * \endcode
 *
 * Where the choices possible for '\c $PRECONDITIONER' are:
 * - \c Jacobi: A Jacobi preconditioner
 * - \c GaussSeidel: A Gauss-Seidel preconditioner
 * - \c SSOR: A symmetric successive overrelaxation (SSOR) preconditioner
 * - \c SOR: A successive overrelaxation (SOR) preconditioner
 * - \c ILUn: An ILU(n) preconditioner
 * - \c ILU0: An ILU(0) preconditioner. The results of this
 *            preconditioner are the same as setting the
 *            PreconditionerOrder property to 0 and using the ILU(n)
 *            preconditioner. The reason for the existence of ILU0 is
 *            that it is computationally cheaper because it does not
 *            need to consider things which are only required for
 *            higher orders
 */
template <class TypeTag>
class FemSolverBackend
{
protected:
    typedef typename GET_PROP_TYPE(TypeTag, LinearSolverBackend) Implementation;

    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, JacobianMatrix) LinearOperator;
    typedef typename GET_PROP_TYPE(TypeTag, GlobalEqVector) Vector;
    typedef typename GET_PROP_TYPE(TypeTag, BorderListCreator) BorderListCreator;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

    typedef typename GET_PROP_TYPE(TypeTag, Overlap) Overlap;
    typedef typename GET_PROP_TYPE(TypeTag, OverlappingVector) OverlappingVector;
    typedef typename GET_PROP_TYPE(TypeTag, OverlappingMatrix) OverlappingMatrix;

    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunctionSpace) DiscreteFunctionSpace;
    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunction)      SolverDiscreteFunction;

    typedef ISTLBlockVectorDiscreteFunction< DiscreteFunctionSpace >
        VectorWrapperDiscreteFunction;

    //typedef typename GET_PROP_TYPE(TypeTag, PreconditionerWrapper) PreconditionerWrapper;
    //typedef typename PreconditionerWrapper::SequentialPreconditioner SequentialPreconditioner;

    enum { dimWorld = GridView::dimensionworld };

public:
    FemSolverBackend(const Simulator& simulator)
        : simulator_(simulator)
        , x_  ()
        , rhs_()
    {
    }

    ~FemSolverBackend()
    { cleanup_(); }

    /*!
     * \brief Register all run-time parameters for the linear solver.
     */
    static void registerParameters()
    {
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, LinearSolverTolerance,
                             "The maximum allowed error between of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, unsigned, LinearSolverOverlapSize,
                             "The size of the algebraic overlap for the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverMaxIterations,
                             "The maximum number of iterations of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverVerbosity,
                             "The verbosity level of the linear solver");

        // PreconditionerWrapper::registerParameters();
    }

    /*!
     * \brief Causes the solve() method to discared the structure of the linear system of
     *        equations the next time it is called.
     */
    void eraseMatrix()
    { cleanup_(); }

    void prepareMatrix(const LinearOperator& linOp)
    {
        Scalar linearSolverTolerance = EWOMS_GET_PARAM(TypeTag, Scalar, LinearSolverTolerance);
        Scalar linearSolverAbsTolerance = this->simulator_.model().newtonMethod().tolerance() / 10.0;

        // reset linear solver
        invOp_.reset( new InverseOperator( linOp, linearSolverTolerance, linearSolverAbsTolerance ) );

        if( ! x_ ) {
            x_.reset( new SolverDiscreteFunction( "FSB::x_", space() ) );
        }

        if( ! rhs_ ) {
            rhs_.reset( new SolverDiscreteFunction( "FSB::rhs_", space() ) );
        }

        // not needed
        asImp_().rescale_();
    }

    void prepareRhs(const LinearOperator& linOp, Vector& b)
    {
        // copy to discrete function
        toDF( b, *rhs_ );

        // not needed ?
        // rhs_.communicate();
    }

    /*!
     * \brief Actually solve the linear system of equations.
     *
     * \return true if the residual reduction could be achieved, else false.
     */
    bool solve(Vector& x)
    {
        // copy to discrete function
        toDF( x, *x_ );

        // solve with right hand side rhs and store in x
        invOp_( *rhs_, *x_ );

        // copy back to solution
        toVec( *x_, x );

        // return the result of the solver
        return true; //result;
    }

protected:
    Implementation& asImp_()
    { return *static_cast<Implementation *>(this); }

    const Implementation& asImp_() const
    { return *static_cast<const Implementation *>(this); }
    }

    const DiscreteFunctionSpace& space() const {
        return simulator_.model().space();
    }

    void toDF( SolutionVector& x, SolverDiscreteFunction& f ) const
    {
        VectorWrapperDiscreteFunction xf( "wrap x", space(), x );
        f.assign( xf );
    }

    void toVec( const SolverDiscreteFunction& f, SolutionVector& x ) const
    {
        VectorWrapperDiscreteFunction xf( "wrap x", space(), x );
        xf.assign( f );
    }

    void rescale_()
    {
        /*
        const auto& overlap = overlappingMatrix_->overlap();
        for (unsigned domesticRowIdx = 0; domesticRowIdx < overlap.numLocal(); ++domesticRowIdx) {
            Index nativeRowIdx = overlap.domesticToNative(static_cast<Index>(domesticRowIdx));
            auto& row = (*overlappingMatrix_)[domesticRowIdx];

            auto colIt = row.begin();
            const auto& colEndIt = row.end();
            for (; colIt != colEndIt; ++ colIt) {
                auto& entry = *colIt;
                for (unsigned i = 0; i < entry.rows; ++i)
                    entry[i] *= simulator_.model().eqWeight(nativeRowIdx, i);
            }

            auto& rhsEntry = (*overlappingb_)[domesticRowIdx];
            for (unsigned i = 0; i < rhsEntry.size(); ++i)
                rhsEntry[i] *= simulator_.model().eqWeight(nativeRowIdx, i);
        }
        */
    }

    void cleanup_()
    {
        invOp_.reset();
        x_.reset();
        rhs_.reset();
    }

    const Simulator& simulator_;

    std::unique_ptr< SolverDiscreteFunction > x_;
    std::unique_ptr< SolverDiscreteFunction > rhs_;
};
}} // namespace Linear, Ewoms

namespace Ewoms {
namespace Properties {
//! make the linear solver shut up by default
SET_INT_PROP(ParallelBaseLinearSolver, LinearSolverVerbosity, 0);

//! set the preconditioner relaxation parameter to 1.0 by default
SET_SCALAR_PROP(ParallelBaseLinearSolver, PreconditionerRelaxation, 1.0);

//! set the preconditioner order to 0 by default
SET_INT_PROP(ParallelBaseLinearSolver, PreconditionerOrder, 0);

//! by default use the same kind of floating point values for the linearization and for
//! the linear solve
SET_TYPE_PROP(ParallelBaseLinearSolver,
              LinearSolverScalar,
              typename GET_PROP_TYPE(TypeTag, Scalar));

SET_PROP(ParallelBaseLinearSolver, OverlappingMatrix)
{
    static constexpr int numEq = GET_PROP_VALUE(TypeTag, NumEq);
    typedef typename GET_PROP_TYPE(TypeTag, LinearSolverScalar) LinearSolverScalar;
    typedef Dune::FieldMatrix<LinearSolverScalar, numEq, numEq> MatrixBlock;
    typedef Dune::BCRSMatrix<MatrixBlock> NonOverlappingMatrix;
    typedef Ewoms::Linear::OverlappingBCRSMatrix<NonOverlappingMatrix> type;
};

SET_TYPE_PROP(ParallelBaseLinearSolver,
              Overlap,
              typename GET_PROP_TYPE(TypeTag, OverlappingMatrix)::Overlap);

SET_PROP(ParallelBaseLinearSolver, OverlappingVector)
{
    static constexpr int numEq = GET_PROP_VALUE(TypeTag, NumEq);
    typedef typename GET_PROP_TYPE(TypeTag, LinearSolverScalar) LinearSolverScalar;
    typedef Dune::FieldVector<LinearSolverScalar, numEq> VectorBlock;
    typedef typename GET_PROP_TYPE(TypeTag, Overlap) Overlap;
    typedef Ewoms::Linear::OverlappingBlockVector<VectorBlock, Overlap> type;
};

SET_PROP(ParallelBaseLinearSolver, OverlappingScalarProduct)
{
    typedef typename GET_PROP_TYPE(TypeTag, OverlappingVector) OverlappingVector;
    typedef typename GET_PROP_TYPE(TypeTag, Overlap) Overlap;
    typedef Ewoms::Linear::OverlappingScalarProduct<OverlappingVector, Overlap> type;
};

SET_PROP(ParallelBaseLinearSolver, OverlappingLinearOperator)
{
    typedef typename GET_PROP_TYPE(TypeTag, OverlappingMatrix) OverlappingMatrix;
    typedef typename GET_PROP_TYPE(TypeTag, OverlappingVector) OverlappingVector;
    typedef Ewoms::Linear::OverlappingOperator<OverlappingMatrix, OverlappingVector,
                                               OverlappingVector> type;
};

#if DUNE_VERSION_NEWER(DUNE_ISTL, 2,7)
SET_TYPE_PROP(ParallelBaseLinearSolver,
              PreconditionerWrapper,
              Ewoms::Linear::PreconditionerWrapperILU<TypeTag>);
#else
SET_TYPE_PROP(ParallelBaseLinearSolver,
              PreconditionerWrapper,
              Ewoms::Linear::PreconditionerWrapperILU0<TypeTag>);
#endif

//! set the default overlap size to 2
SET_INT_PROP(ParallelBaseLinearSolver, LinearSolverOverlapSize, 2);

//! set the default number of maximum iterations for the linear solver
SET_INT_PROP(ParallelBaseLinearSolver, LinearSolverMaxIterations, 1000);
} // namespace Properties
} // namespace Ewoms

#endif // HAVE_DUNE_FEM

#endif

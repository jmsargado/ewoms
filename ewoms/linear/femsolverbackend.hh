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

#include <ewoms/disc/common/fvbaseproperties.hh>

#if HAVE_DUNE_FEM

#define DISABLE_AMG_DIRECTSOLVER 1
#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/petscsolver.hh>

#include <ewoms/common/genericguard.hh>
#include <ewoms/common/propertysystem.hh>
#include <ewoms/common/parametersystem.hh>

#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/common/fvector.hh>


#include <ewoms/linear/parallelbicgstabbackend.hh>

#include <sstream>
#include <memory>
#include <iostream>

namespace Ewoms {
namespace Linear {
template <class TypeTag>
class FemSolverBackend;
}} // namespace Linear, Ewoms


namespace Ewoms {
namespace Properties {
NEW_TYPE_TAG(FemSolverBackend);

SET_TYPE_PROP(FemSolverBackend,
              LinearSolverBackend,
              Ewoms::Linear::FemSolverBackend<TypeTag>);

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
    typedef typename GET_PROP_TYPE(TypeTag, LinearOperator) LinearOperator;
    typedef typename GET_PROP_TYPE(TypeTag, GlobalEqVector) Vector;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunctionSpace) DiscreteFunctionSpace;
    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunction)      SolverDiscreteFunction;

    // discrete function to wrap what is used as Vector in eWoms
    typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DiscreteFunctionSpace >
        VectorWrapperDiscreteFunction;

    typedef Dune::Fem::ISTLBICGSTABOp< SolverDiscreteFunction, LinearOperator >  InverseLinearOperator;
    //typedef Dune::Fem::PetscInverseOperator< SolverDiscreteFunction, LinearOperator >  InverseLinearOperator;
    //typedef Dune::Fem::OEMBICGSTABOp< SolverDiscreteFunction, LinearOperator >  InverseLinearOperator;

    enum { dimWorld = GridView::dimensionworld };

public:
    FemSolverBackend(const Simulator& simulator)
        : simulator_(simulator)
        , invOp_()
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
        /*
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, LinearSolverTolerance,
                             "The maximum allowed error between of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverMaxIterations,
                             "The maximum number of iterations of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverVerbosity,
                             "The verbosity level of the linear solver");


                             */
        // PreconditionerWrapper::registerParameters();

        // set ilu preconditioner
        Dune::Fem::Parameter::append("istl.preconditioning.method", "ilu" );
        Dune::Fem::Parameter::append("istl.preconditioning.iterations", "0" );
    }

    /*!
     * \brief Causes the solve() method to discared the structure of the linear system of
     *        equations the next time it is called.
     */
    void eraseMatrix()
    { cleanup_(); }

    void prepareMatrix(const LinearOperator& linOp)
    {
        Scalar linearSolverTolerance = 0.01;//EWOMS_GET_PARAM(TypeTag, Scalar, LinearSolverTolerance);
        Scalar linearSolverAbsTolerance = this->simulator_.model().newtonMethod().tolerance() / 10.0;

        // reset linear solver
        LinearOperator& op = const_cast< LinearOperator& > (linOp);
        invOp_.reset( new InverseLinearOperator( op, linearSolverTolerance, linearSolverAbsTolerance ) );

        // not needed
        asImp_().rescale_();
    }

    void prepareRhs(const LinearOperator& linOp, Vector& b)
    {
        if( ! rhs_ ) {
            rhs_.reset( new SolverDiscreteFunction( "FSB::rhs_", space() ) );
        }

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
        if( ! x_ ) {
            x_.reset( new SolverDiscreteFunction( "FSB::x_", space() ) );
        }

        // copy to discrete function
        toDF( x, *x_ );

        // solve with right hand side rhs and store in x
        (*invOp_)( *rhs_, *x_ );

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

    const DiscreteFunctionSpace& space() const {
        return simulator_.model().space();
    }

    void toDF( Vector& x, SolverDiscreteFunction& f ) const
    {
        VectorWrapperDiscreteFunction xf( "wrap x", space(), x );
        f.assign( xf );
    }

    void toVec( const SolverDiscreteFunction& f, Vector& x ) const
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

    std::unique_ptr< InverseLinearOperator > invOp_;

    std::unique_ptr< SolverDiscreteFunction > x_;
    std::unique_ptr< SolverDiscreteFunction > rhs_;
};
}} // namespace Linear, Ewoms

#endif // HAVE_DUNE_FEM

#endif

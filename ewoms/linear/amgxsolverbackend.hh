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
 * \copydoc Ewoms::Linear::AmgXSolverBackend
 */
#ifndef EWOMS_AMGX_SOLVER_BACKEND_HH
#define EWOMS_AMGX_SOLVER_BACKEND_HH

#include <ewoms/disc/common/fvbaseproperties.hh>

#if USE_AMGX_SOLVERS

#if ! HAVE_PETSC
#error "PETSc is needed for the AMGX solver backend"
#endif

#define DISABLE_AMG_DIRECTSOLVER 1
#include <dune/fem/solver/istlsolver.hh>
#include <dune/fem/solver/petscsolver.hh>
#include <dune/fem/solver/krylovinverseoperators.hh>
#include <dune/fem/function/petscdiscretefunction.hh>
#include <ewoms/common/genericguard.hh>
#include <ewoms/common/propertysystem.hh>
#include <ewoms/common/parametersystem.hh>

#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/common/fvector.hh>


#include <ewoms/linear/parallelbicgstabbackend.hh>
#include <ewoms/linear/istlsolverwrappers.hh>

#include <sstream>
#include <memory>
#include <iostream>

namespace Ewoms {
namespace Linear {
template <class TypeTag>
class AmgXSolverBackend;
}} // namespace Linear, Ewoms


BEGIN_PROPERTIES

NEW_TYPE_TAG(AmgXSolverBackend);

SET_TYPE_PROP(AmgXSolverBackend,
              LinearSolverBackend,
              Ewoms::Linear::AmgXSolverBackend<TypeTag>);

//NEW_PROP_TAG(LinearSolverTolerance);
NEW_PROP_TAG(LinearSolverMaxIterations);
NEW_PROP_TAG(LinearSolverVerbosity);
NEW_PROP_TAG(LinearSolverMaxError);
NEW_PROP_TAG(LinearSolverOverlapSize);
//! The order of the sequential preconditioner
NEW_PROP_TAG(PreconditionerOrder);

//! The relaxation factor of the preconditioner
NEW_PROP_TAG(PreconditionerRelaxation);

//! make the linear solver shut up by default
SET_INT_PROP(AmgXSolverBackend, LinearSolverVerbosity, 0);

//! set the default number of maximum iterations for the linear solver
SET_INT_PROP(AmgXSolverBackend, LinearSolverMaxIterations, 1000);

SET_SCALAR_PROP(AmgXSolverBackend, LinearSolverMaxError, 1e7);

//! set the default overlap size to 2
SET_INT_PROP(AmgXSolverBackend, LinearSolverOverlapSize, 2);

//! set the preconditioner order to 0 by default
SET_INT_PROP(AmgXSolverBackend, PreconditionerOrder, 0);

//! set the preconditioner relaxation parameter to 1.0 by default
SET_SCALAR_PROP(AmgXSolverBackend, PreconditionerRelaxation, 1.0);

//! make the linear solver shut up by default
//SET_SCALAR_PROP(AmgXSolverBackend, LinearSolverTolerance, 0.01);

END_PROPERTIES

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
class AmgXSolverBackend
{
protected:
    typedef typename GET_PROP_TYPE(TypeTag, LinearSolverBackend) Implementation;

    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, JacobianMatrix) LinearOperator;
    typedef typename GET_PROP_TYPE(TypeTag, GlobalEqVector) Vector;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunctionSpace) DiscreteFunctionSpace;
    typedef typename GET_PROP_TYPE(TypeTag, DiscreteFunction)      DiscreteFunction;

    // discrete function to wrap what is used as Vector in eWoms
    typedef Dune::Fem::ISTLBlockVectorDiscreteFunction< DiscreteFunctionSpace >
        VectorWrapperDiscreteFunction;
    typedef Dune::Fem::PetscDiscreteFunction< DiscreteFunctionSpace >
        PetscDiscreteFunctionType;

    enum { dimWorld = GridView::dimensionworld };

public:
    AmgXSolverBackend(const Simulator& simulator)
        : simulator_(simulator)
        , amgxSolver_()
        , rhs_( nullptr )
    {
    }

    ~AmgXSolverBackend()
    { cleanup_(); }

    /*!
     * \brief Register all run-time parameters for the linear solver.
     */
    static void registerParameters()
    {
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, LinearSolverTolerance,
                             "The maximum allowed error between of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverMaxIterations,
                             "The maximum number of iterations of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, int, LinearSolverVerbosity,
                             "The verbosity level of the linear solver");
        EWOMS_REGISTER_PARAM(TypeTag, unsigned, LinearSolverOverlapSize,
                             "The size of the algebraic overlap for the linear solver");

        EWOMS_REGISTER_PARAM(TypeTag, Scalar, LinearSolverMaxError,
                             "The maximum residual error which the linear solver tolerates"
                             " without giving up");

        EWOMS_REGISTER_PARAM(TypeTag, int, PreconditionerOrder,
                             "The order of the preconditioner");
        EWOMS_REGISTER_PARAM(TypeTag, Scalar, PreconditionerRelaxation,
                             "The relaxation factor of the preconditioner");


        //PreconditionerWrapper::registerParameters();

        // set ilu preconditioner istl
        Dune::Fem::Parameter::append("istl.preconditioning.method", "ilu" );
        Dune::Fem::Parameter::append("istl.preconditioning.relaxation", "0.9" );
        Dune::Fem::Parameter::append("istl.preconditioning.iterations", "0" );
        Dune::Fem::Parameter::append("fem.solver.errormeasure", "residualreduction" );

        // possible solvers: cg, bicg, bicgstab, gmres
        Dune::Fem::Parameter::append("petsc.kspsolver.method", "bicgstab" );
        // possible precond: none, asm, sor, jacobi, hypre, ilu-n, lu, icc ml superlu mumps
        Dune::Fem::Parameter::append("petsc.preconditioning.method", "ilu");

        //int verbosity = EWOMS_GET_PARAM(TypeTag, int, LinearSolverVerbosity);
        //if( verbosity )
        //    Dune::Fem::Parameter::append("fem.solver.verbose", "true" );
        //else
        Dune::Fem::Parameter::append("fem.solver.verbose", "true" );
    }

    /*!
     * \brief Causes the solve() method to discared the structure of the linear system of
     *        equations the next time it is called.
     */
    void eraseMatrix()
    { cleanup_(); }

    void prepareMatrix(const LinearOperator& op)
    {
        Scalar linearSolverTolerance = EWOMS_GET_PARAM(TypeTag, Scalar, LinearSolverTolerance);
        Scalar linearSolverAbsTolerance = this->simulator_.model().newtonMethod().tolerance() / 100000.0;

        // reset linear solver
        std::string mode = "AmgX_GPU";
        std::string solverconfig = "./";
        amgxSolver.initialize(MPI_COMM_WORLD, mode, solverconfig);
    }

    void prepareRhs(const LinearOperator& linOp, Vector& b)
    {
        rhs_ = &b;
    }

    /*!
     * \brief Actually solve the linear system of equations.
     *
     * \return true if the residual reduction could be achieved, else false.
     */
    bool solve(Vector& x)
    {
        // wrap x into discrete function X (no copy)
        VectorWrapperDiscreteFunction X( "FSB::x",   space(), x );
        VectorWrapperDiscreteFunction B( "FSB::rhs", space(), *rhs_ );

        if( ! petscRhs_ )
        {
            petscRhs_.reset( new PetscDiscreteFunctionType( "AMGX::rhs", space() ) );
        }
        if( ! petscX_ )
        {
            petscX_.reset( new PetscDiscreteFunctionType("AMGX::X", space()) );
        }

        petscRhs_->assign( B );
        petscX_->clear();

        // solve with right hand side rhs and store in x
        amgxSolver_.solve( petcsX_->petscVector() , petcsRhs_->petscVector() );

        // copy result to ewoms solution
        X.assign( *petscX_ );

        int iters;
        amgxSolver_.getIters(iters);

        // return the result of the solver
        return true;
    }

    /*!
     * \brief Return number of iterations used during last solve.
     */
    size_t iterations () const {
        //assert( amgxSolver_);
        return 10; //std::abs(amgxSolver_->iterations());
    }

protected:
    Implementation& asImp_()
    { return *static_cast<Implementation *>(this); }

    const Implementation& asImp_() const
    { return *static_cast<const Implementation *>(this); }

    const DiscreteFunctionSpace& space() const {
        return simulator_.model().space();
    }

    void cleanup_()
    {
        //amgxSolver_.reset();
        amgxSolver_.finalize();
        rhs_ = nullptr;

        petscRhs_.reset();
        petscX_.reset();
    }

    const Simulator& simulator_;

    std::unique_ptr< PetscDiscreteFunctionType > petscRhs_;
    std::unique_ptr< PetscDiscreteFunctionType > petscX_;

    AmgXSolver amgxSolver_;

    Vector* rhs_;
};
}} // namespace Linear, Ewoms

#endif // HAVE_DUNE_FEM

#endif

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
 * \copydoc Ewoms::Linear::ISTLMatrixBackend
 */
#ifndef EWOMS_ISTL_MATRIX_BACKEND_HH
#define EWOMS_ISTL_MATRIX_BACKEND_HH

#include <dune/istl/bcrsmatrix.hh>
#include <dune/common/fmatrix.hh>
#include <dune/common/version.hh>

#include <ewoms/linear/matrixblock.hh>

namespace Ewoms {
namespace Linear {

/*!
 * \ingroup Linear
 * \brief A linear solver backend for the SuperLU sparse matrix library.
 */
template<class Block, class A=std::allocator< Block > >
class ISTLMatrixBackend
{
    typedef A AllocatorType;
public:
    typedef Dune::BCRSMatrix< Block, AllocatorType >   MatrixType;
    // block_type is the same as Block
    typedef typename MatrixType :: block_type          block_type;

    ISTLMatrixBackend( const std::string& name, const size_t rows, const size_t columns )
        : rows_( rows )
        , columns_( columns )
        , matrix_()
    {}

    explicit ISTLMatrixBackend( const std::string& name, const int rows, const int columns )
        : ISTLMatrixBackend( name, size_t(rows), size_t(columns) )
    {}

    template <class DomainSpace, class RangeSpace>
    ISTLMatrixBackend( const std::string& name, const DomainSpace& domainSpace, const RangeSpace& rangeSpace )
        : ISTLMatrixBackend( name, domainSpace.size()/DomainSpace::dimRange, rangeSpace.size()/RangeSpace::dimRange )
    {
    }

    // allocate matrix from given sparsity pattern
    template <class Set>
    inline void reserve( const std::vector< Set >& sparsityPattern )
    {
        // allocate raw matrix
        matrix_.reset( new MatrixType(rows_, columns_, MatrixType::random) );

        // make sure sparsityPattern is consistent with number of rows
        assert( rows_ == sparsityPattern.size() );

        // allocate space for the rows of the matrix
        for (size_t dofIdx = 0; dofIdx < rows_; ++ dofIdx)
        {
            matrix_->setrowsize(dofIdx, sparsityPattern[dofIdx].size());
        }

        matrix_->endrowsizes();

        // fill the rows with indices. each degree of freedom talks to
        // all of its neighbors. (it also talks to itself since
        // degrees of freedom are sometimes quite egocentric.)
        for (size_t dofIdx = 0; dofIdx < rows_; ++ dofIdx)
        {
            auto nIt    = sparsityPattern[dofIdx].begin();
            auto nEndIt = sparsityPattern[dofIdx].end();
            for (; nIt != nEndIt; ++nIt)
            {
                matrix_->addindex(dofIdx, *nIt);
            }
        }
        matrix_->endindices();
    }

    inline MatrixType& matrix() { return *matrix_; }
    inline const MatrixType& matrix() const { return *matrix_; }

    inline size_t N () const { return rows_; }
    inline size_t M () const { return columns_; }

    // set all matrix entries to zero
    inline void clear()
    {
        (*matrix_) = typename block_type :: field_type(0);
    }

    inline void unitRow( const size_t row )
    {
        block_type idBlock( 0 );
        for (int i = 0; i < idBlock.rows; ++i)
            idBlock[i][i] = 1.0;

        auto& matRow = (*matrix_)[ row ];
        auto colIt = matRow.begin();
        const auto& colEndIt = matRow.end();
        for (; colIt != colEndIt; ++colIt)
        {
            if( colIt.index() == row )
                *colIt = idBlock;
            else
                *colIt = 0.0;
        }
    }

    inline void getBlock( const size_t row, const size_t col, block_type& entry ) const
    {
        entry = (*matrix_)[ row ][ col ];
    }

    inline void setBlock( const size_t row, const size_t col, const block_type& entry )
    {
        (*matrix_)[ row ][ col ] = entry;
    }

    inline void addBlock( const size_t row, const size_t col, const block_type& entry )
    {
        (*matrix_)[ row ][ col ] += entry;
    }

    inline void communicate()
    {
        // nothing to do here
        // may call compress when implicit build mode is used
    }

protected:
    size_t rows_;
    size_t columns_;

    std::unique_ptr< MatrixType > matrix_;
};

} // namespace Linear
} // namespace Ewoms

#endif

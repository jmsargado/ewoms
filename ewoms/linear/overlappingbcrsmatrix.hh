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
 * \copydoc Ewoms::Linear::OverlappingBCRSMatrix
 */
#ifndef EWOMS_OVERLAPPING_BCRS_MATRIX_HH
#define EWOMS_OVERLAPPING_BCRS_MATRIX_HH

#include <ewoms/linear/domesticoverlapfrombcrsmatrix.hh>
#include <ewoms/linear/globalindices.hh>
#include <ewoms/linear/blacklist.hh>
#include <ewoms/parallel/mpibuffer.hh>

#include <opm/material/common/Valgrind.hpp>

#include <dune/istl/scalarproducts.hh>
#include <dune/istl/io.hh>

#include <algorithm>
#include <set>
#include <map>
#include <iostream>
#include <vector>
#include <memory>

namespace Ewoms {
namespace Linear {

/*!
 * \brief An overlap aware block-compressed row storage (BCRS) matrix.
 */
template <class BCRSMatrix>
class OverlappingBCRSMatrix : public BCRSMatrix
{
    typedef BCRSMatrix ParentType;

public:
    typedef Ewoms::Linear::DomesticOverlapFromBCRSMatrix Overlap;

private:
    typedef std::vector<std::set<Index> > Entries;

public:
    typedef typename ParentType::ColIterator ColIterator;
    typedef typename ParentType::ConstColIterator ConstColIterator;
    typedef typename ParentType::block_type block_type;
    typedef typename ParentType::field_type field_type;

    // no real copying done at the moment
    OverlappingBCRSMatrix(const OverlappingBCRSMatrix& other)
        : ParentType(other)
    {}

    template <class JacobianMatrix>
    OverlappingBCRSMatrix(const JacobianMatrix& jacobian,
                          const BorderList& borderList,
                          const BlackList& blackList,
                          unsigned overlapSize)
        : OverlappingBCRSMatrix( jacobian.matrix(), borderList, blackList, overlapSize )
    {}

    OverlappingBCRSMatrix(const ParentType& nativeMatrix,
                          const BorderList& borderList,
                          const BlackList& blackList,
                          unsigned overlapSize)
    {
        overlap_ = std::make_shared<Overlap>(nativeMatrix, borderList, blackList, overlapSize);
        myRank_ = 0;
#if HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank_);
#endif // HAVE_MPI

        // build the overlapping matrix from the non-overlapping
        // matrix and the overlap
        build_(nativeMatrix);
    }

    // this constructor is required to make the class compatible with the SeqILU class of
    // Dune >= 2.7.
    OverlappingBCRSMatrix(size_t numRows,
                          size_t numCols,
                          typename BCRSMatrix::BuildMode buildMode)
    { throw std::logic_error("OverlappingBCRSMatrix objects cannot be build from scratch!"); }

    ~OverlappingBCRSMatrix()
    {
        if (overlap_.use_count() == 0)
            return;

        // delete all MPI buffers
        const PeerSet& peerSet = overlap_->peerSet();
        typename PeerSet::const_iterator peerIt = peerSet.begin();
        typename PeerSet::const_iterator peerEndIt = peerSet.end();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            delete rowSizesRecvBuff_[peerRank];
            delete rowIndicesRecvBuff_[peerRank];
            delete entryColIndicesRecvBuff_[peerRank];
            delete entryValuesRecvBuff_[peerRank];

            delete numRowsSendBuff_[peerRank];
            delete rowSizesSendBuff_[peerRank];
            delete rowIndicesSendBuff_[peerRank];
            delete entryColIndicesSendBuff_[peerRank];
            delete entryValuesSendBuff_[peerRank];
        }
    }

    ParentType& asParent()
    { return *this; }

    const ParentType& asParent() const
    { return *this; }

    /*!
     * \brief Returns the domestic overlap for the process.
     */
    const Overlap& overlap() const
    { return *overlap_; }

    /*!
     * \brief Assign and syncronize the overlapping matrix from a non-overlapping one.
     */
    template <class JacobianMatrix>
    void assignAdd(const JacobianMatrix& jacobian)
    {
        assignAdd( jacobian.matrix() );
    }

    /*!
     * \brief Assign and syncronize the overlapping matrix from a non-overlapping one.
     */
    void assignAdd(const ParentType& nativeMatrix)
    {
        // copy the native entries
        assignFromNative(nativeMatrix);

        // communicate and add the contents of overlapping rows
        syncAdd();
    }

    /*!
     * \brief Assign and syncronize the overlapping matrix from a
     *       non-overlapping one.
     *
     * The non-master entries are copied from the master
     */
    template <class JacobianMatrix>
    void assignCopy(const JacobianMatrix& jacobian)
    {
        assignCopy( jacobian.matrix() );
    }


    /*!
     * \brief Assign and syncronize the overlapping matrix from a
     *       non-overlapping one.
     *
     * The non-master entries are copied from the master
     */
    void assignCopy(const ParentType& nativeMatrix)
    {
        // copy the native entries
        assignFromNative(nativeMatrix);

        // communicate and add the contents of overlapping rows
        syncCopy();
    }

    /*!
     * \brief Set the identity matrix on the main diagonal of front indices.
     */
    void resetFront()
    {
        // create an identity matrix
        block_type idMatrix(0.0);
        for (unsigned i = 0; i < idMatrix.size(); ++i)
            idMatrix[i][i] = 1.0;

        int numLocal = overlap_->numLocal();
        int numDomestic = overlap_->numDomestic();
        for (int domRowIdx = numLocal; domRowIdx < numDomestic; ++domRowIdx) {
            if (overlap_->isFront(domRowIdx)) {
                // set the front rows to a diagonal 1
                (*this)[domRowIdx] = 0.0;
                (*this)[domRowIdx][domRowIdx] = idMatrix;
            }
        }
    }

    void print() const
    {
        overlap_->print();

        for (int i = 0; i < this->N(); ++i) {
            if (overlap_->isLocal(i))
                std::cout << " ";
            else
                std::cout << "*";
            std::cout << "row " << i << " ";

            typedef typename BCRSMatrix::ConstColIterator ColIt;
            ColIt colIt = (*this)[i].begin();
            ColIt colEndIt = (*this)[i].end();
            for (int j = 0; j < this->M(); ++j) {
                if (colIt != colEndIt && j == colIt.index()) {
                    ++colIt;
                    if (overlap_->isBorder(j))
                        std::cout << "|";
                    else if (overlap_->isLocal(j))
                        std::cout << "X";
                    else
                        std::cout << "*";
                }
                else
                    std::cout << " ";
            }
            std::cout << "\n" << std::flush;
        }
        Dune::printSparseMatrix(std::cout,
                                *static_cast<const BCRSMatrix *>(this),
                                "M",
                                "row");
    }

    template <class JacobianMatrix>
    void assignFromNative(const JacobianMatrix& jacobian)
    {
        assignFromNative( jacobian.matrix() );
    }

    void assignFromNative(const ParentType& nativeMatrix)
    {
        // first, set everything to 0,
        BCRSMatrix::operator=(0.0);

        // then copy the domestic entries of the native matrix to the overlapping matrix
        for (unsigned nativeRowIdx = 0; nativeRowIdx < nativeMatrix.N(); ++nativeRowIdx) {
            Index domesticRowIdx = overlap_->nativeToDomestic(static_cast<Index>(nativeRowIdx));
            if (domesticRowIdx < 0) {
                continue; // row corresponds to a black-listed entry
            }

            auto nativeColIt = nativeMatrix[nativeRowIdx].begin();
            const auto& nativeColEndIt = nativeMatrix[nativeRowIdx].end();
            for (; nativeColIt != nativeColEndIt; ++nativeColIt) {
                Index domesticColIdx = overlap_->nativeToDomestic(static_cast<Index>(nativeColIt.index()));

                // make sure to include all off-diagonal entries, even those which belong
                // to DOFs which are managed by a peer process. For this, we have to
                // re-map the column index of the black-listed index to a native one.
                if (domesticColIdx < 0)
                    domesticColIdx = overlap_->blackList().nativeToDomestic(static_cast<Index>(nativeColIt.index()));

                if (domesticColIdx < 0)
                    // there is no domestic index which corresponds to a black-listed
                    // one. this can happen if the grid overlap is larger than the
                    // algebraic one...
                    continue;

                // we need to copy the block matrices manually since it seems that (at
                // least some versions of) Dune have an endless recursion bug when
                // assigning dense matrices of different field type
                const auto& src = *nativeColIt;
                auto& dest = (*this)[static_cast<unsigned>(domesticRowIdx)][static_cast<unsigned>(domesticColIdx)];
                for (unsigned i = 0; i < src.rows; ++i) {
                    for (unsigned j = 0; j < src.cols; ++j) {
                        dest[i][j] = static_cast<field_type>(src[i][j]);
                    }
                }
            }
        }
    }

    // communicates and adds up the contents of overlapping rows
    void syncAdd()
    {
        // first, send all entries to the peers
        const PeerSet& peerSet = overlap_->peerSet();
        typename PeerSet::const_iterator peerIt = peerSet.begin();
        typename PeerSet::const_iterator peerEndIt = peerSet.end();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            sendEntries_(peerRank);
        }

        // then, receive entries from the peers
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            receiveAddEntries_(peerRank);
        }

        // finally, make sure that everything which we send was
        // received by the peers
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;
            entryValuesSendBuff_[peerRank]->wait();
        }
    }

    // communicates and copies the contents of overlapping rows from
    // the master
    void syncCopy()
    {
        // first, send all entries to the peers
        const PeerSet& peerSet = overlap_->peerSet();
        typename PeerSet::const_iterator peerIt = peerSet.begin();
        typename PeerSet::const_iterator peerEndIt = peerSet.end();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            sendEntries_(peerRank);
        }

        // then, receive entries from the peers
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            receiveCopyEntries_(peerRank);
        }

        // finally, make sure that everything which we send was
        // received by the peers
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;
            entryValuesSendBuff_[peerRank]->wait();
        }
    }

private:
    template <class NativeBCRSMatrix>
    void build_(const NativeBCRSMatrix& nativeMatrix)
    {
        size_t numDomestic = overlap_->numDomestic();

        // allocate the rows
        this->setSize(numDomestic, numDomestic);
        this->setBuildMode(ParentType::random);

        // communicate the entries
        buildIndices_(nativeMatrix);
    }

    template <class NativeBCRSMatrix>
    void buildIndices_(const NativeBCRSMatrix& nativeMatrix)
    {
        /////////
        // first, add all local matrix entries
        /////////
        entries_.resize(overlap_->numDomestic());
        for (unsigned nativeRowIdx = 0; nativeRowIdx < nativeMatrix.N(); ++nativeRowIdx) {
            int domesticRowIdx = overlap_->nativeToDomestic(static_cast<Index>(nativeRowIdx));
            if (domesticRowIdx < 0)
                continue;

            auto nativeColIt = nativeMatrix[nativeRowIdx].begin();
            const auto& nativeColEndIt = nativeMatrix[nativeRowIdx].end();
            for (; nativeColIt != nativeColEndIt; ++nativeColIt) {
                int domesticColIdx = overlap_->nativeToDomestic(static_cast<Index>(nativeColIt.index()));

                // make sure to include all off-diagonal entries, even those which belong
                // to DOFs which are managed by a peer process. For this, we have to
                // re-map the column index of the black-listed index to a native one.
                if (domesticColIdx < 0) {
                    domesticColIdx = overlap_->blackList().nativeToDomestic(static_cast<Index>(nativeColIt.index()));
                }

                if (domesticColIdx < 0)
                    continue;

                entries_[static_cast<unsigned>(domesticRowIdx)].insert(domesticColIdx);
            }
        }

        /////////
        // add the indices for all additional entries
        /////////

        // first, send all our indices to all peers
        const PeerSet& peerSet = overlap_->peerSet();
        typename PeerSet::const_iterator peerIt = peerSet.begin();
        typename PeerSet::const_iterator peerEndIt = peerSet.end();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;
            sendIndices_(nativeMatrix, peerRank);
        }

        // then recieve all indices from the peers
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;
            receiveIndices_(peerRank);
        }

        // wait until all send operations are completed
        peerIt = peerSet.begin();
        for (; peerIt != peerEndIt; ++peerIt) {
            ProcessRank peerRank = *peerIt;

            numRowsSendBuff_[peerRank]->wait();
            rowSizesSendBuff_[peerRank]->wait();
            rowIndicesSendBuff_[peerRank]->wait();
            entryColIndicesSendBuff_[peerRank]->wait();

            // convert the global indices in the send buffers to domestic
            // ones
            globalToDomesticBuff_(*rowIndicesSendBuff_[peerRank]);
            globalToDomesticBuff_(*entryColIndicesSendBuff_[peerRank]);
        }

        /////////
        // actually initialize the BCRS matrix structure
        /////////

        // set the row sizes
        size_t numDomestic = overlap_->numDomestic();
        for (unsigned rowIdx = 0; rowIdx < numDomestic; ++rowIdx) {
            unsigned numCols = 0;
            const auto& colIndices = entries_[rowIdx];
            auto colIdxIt = colIndices.begin();
            const auto& colIdxEndIt = colIndices.end();
            for (; colIdxIt != colIdxEndIt; ++colIdxIt) {
                if (*colIdxIt < 0)
                    // the matrix for the local process does not know about this DOF
                    continue;

                ++numCols;
            }

            this->setrowsize(rowIdx, numCols);
        }
        this->endrowsizes();

        // set the indices
        for (unsigned rowIdx = 0; rowIdx < numDomestic; ++rowIdx) {
            const auto& colIndices = entries_[rowIdx];

            auto colIdxIt = colIndices.begin();
            const auto& colIdxEndIt = colIndices.end();
            for (; colIdxIt != colIdxEndIt; ++colIdxIt) {
                if (*colIdxIt < 0)
                    // the matrix for the local process does not know about this DOF
                    continue;

                this->addindex(rowIdx, static_cast<unsigned>(*colIdxIt));
            }
        }
        this->endindices();

        // free the memory occupied by the array of the matrix entries
        entries_.clear();
    }

    // send the overlap indices to a peer
    template <class NativeBCRSMatrix>
    void sendIndices_(const NativeBCRSMatrix& nativeMatrix, ProcessRank peerRank)
    {
#if HAVE_MPI
        // send size of foreign overlap to peer
        size_t numOverlapRows = overlap_->foreignOverlapSize(peerRank);
        numRowsSendBuff_[peerRank] = new MpiBuffer<unsigned>(1);
        (*numRowsSendBuff_[peerRank])[0] = static_cast<unsigned>(numOverlapRows);
        numRowsSendBuff_[peerRank]->send(peerRank);

        // allocate the buffers which hold the global indices of each row and the number
        // of entries which need to be communicated by the respective row
        rowIndicesSendBuff_[peerRank] = new MpiBuffer<Index>(numOverlapRows);
        rowSizesSendBuff_[peerRank] = new MpiBuffer<unsigned>(numOverlapRows);

        // compute the sets of the indices of the entries which need to be send to the peer
        typedef std::set<int> ColumnIndexSet;
        typedef std::map<int, ColumnIndexSet> EntryTuples;

        EntryTuples entryIndices;
        unsigned numEntries = 0; // <- total number of matrix entries to be send to the peer
        for (unsigned overlapOffset = 0; overlapOffset < numOverlapRows; ++overlapOffset) {
            Index domesticRowIdx = overlap_->foreignOverlapOffsetToDomesticIdx(peerRank, overlapOffset);
            Index nativeRowIdx = overlap_->domesticToNative(domesticRowIdx);
            Index globalRowIdx = overlap_->domesticToGlobal(domesticRowIdx);

            ColumnIndexSet& colIndices = entryIndices[globalRowIdx];

            auto nativeColIt = nativeMatrix[static_cast<unsigned>(nativeRowIdx)].begin();
            const auto& nativeColEndIt = nativeMatrix[static_cast<unsigned>(nativeRowIdx)].end();
            for (; nativeColIt != nativeColEndIt; ++nativeColIt) {
                unsigned nativeColIdx = static_cast<unsigned>(nativeColIt.index());
                Index domesticColIdx = overlap_->nativeToDomestic(static_cast<Index>(nativeColIdx));

                if (domesticColIdx < 0)
                    // the native column index may be blacklisted, use the corresponding
                    // index in the domestic overlap.
                    domesticColIdx = overlap_->blackList().nativeToDomestic(static_cast<Index>(nativeColIdx));

                if (domesticColIdx < 0)
                    // the column may still not be known locally, i.e. the corresponding
                    // DOF of the row is at the process's front. we don't need this
                    // entry.
                    continue;

                Index globalColIdx = overlap_->domesticToGlobal(domesticColIdx);
                colIndices.insert(globalColIdx);
                ++numEntries;
            }
        };

        // fill the send buffers
        entryColIndicesSendBuff_[peerRank] = new MpiBuffer<Index>(numEntries);
        Index overlapEntryIdx = 0;
        for (unsigned overlapOffset = 0; overlapOffset < numOverlapRows; ++overlapOffset) {
            Index domesticRowIdx = overlap_->foreignOverlapOffsetToDomesticIdx(peerRank, overlapOffset);
            Index globalRowIdx = overlap_->domesticToGlobal(domesticRowIdx);

            (*rowIndicesSendBuff_[peerRank])[overlapOffset] = globalRowIdx;

            const ColumnIndexSet& colIndexSet = entryIndices[globalRowIdx];
            auto* rssb = rowSizesSendBuff_[peerRank];
            (*rssb)[overlapOffset] = static_cast<unsigned>(colIndexSet.size());
            for (auto it = colIndexSet.begin(); it != colIndexSet.end(); ++it) {
                int globalColIdx = *it;

                (*entryColIndicesSendBuff_[peerRank])[static_cast<unsigned>(overlapEntryIdx)] = globalColIdx;
                ++ overlapEntryIdx;
            }
        }

        // actually communicate with the peer
        rowSizesSendBuff_[peerRank]->send(peerRank);
        rowIndicesSendBuff_[peerRank]->send(peerRank);
        entryColIndicesSendBuff_[peerRank]->send(peerRank);

        // create the send buffers for the values of the matrix
        // entries
        entryValuesSendBuff_[peerRank] = new MpiBuffer<block_type>(numEntries);
#endif // HAVE_MPI
    }

    // receive the overlap indices to a peer
    void receiveIndices_(ProcessRank peerRank)
    {
#if HAVE_MPI
        // receive size of foreign overlap to peer
        unsigned numOverlapRows;
        auto& numRowsRecvBuff = numRowsRecvBuff_[peerRank];
        numRowsRecvBuff.resize(1);
        numRowsRecvBuff.receive(peerRank);
        numOverlapRows = numRowsRecvBuff[0];

        // create receive buffer for the row sizes and receive them
        // from the peer
        rowSizesRecvBuff_[peerRank] = new MpiBuffer<unsigned>(numOverlapRows);
        rowIndicesRecvBuff_[peerRank] = new MpiBuffer<Index>(numOverlapRows);
        rowSizesRecvBuff_[peerRank]->receive(peerRank);
        rowIndicesRecvBuff_[peerRank]->receive(peerRank);

        // calculate the total number of indices which are send by the
        // peer
        unsigned totalIndices = 0;
        for (unsigned i = 0; i < numOverlapRows; ++i)
            totalIndices += (*rowSizesRecvBuff_[peerRank])[i];

        // create the buffer to store the column indices of the matrix entries
        entryColIndicesRecvBuff_[peerRank] = new MpiBuffer<Index>(totalIndices);
        entryValuesRecvBuff_[peerRank] = new MpiBuffer<block_type>(totalIndices);

        // communicate with the peer
        entryColIndicesRecvBuff_[peerRank]->receive(peerRank);

        // convert the global indices in the receive buffers to
        // domestic ones
        globalToDomesticBuff_(*rowIndicesRecvBuff_[peerRank]);
        globalToDomesticBuff_(*entryColIndicesRecvBuff_[peerRank]);

        // add the entries to the global entry map
        unsigned k = 0;
        for (unsigned i = 0; i < numOverlapRows; ++i) {
            Index domRowIdx = (*rowIndicesRecvBuff_[peerRank])[i];
            for (unsigned j = 0; j < (*rowSizesRecvBuff_[peerRank])[i]; ++j) {
                Index domColIdx = (*entryColIndicesRecvBuff_[peerRank])[k];
                entries_[static_cast<unsigned>(domRowIdx)].insert(domColIdx);
                ++k;
            }
        }
#endif // HAVE_MPI
    }

    void sendEntries_(ProcessRank peerRank)
    {
#if HAVE_MPI
        auto &mpiSendBuff = *entryValuesSendBuff_[peerRank];

        auto &mpiRowIndicesSendBuff = *rowIndicesSendBuff_[peerRank];
        auto &mpiRowSizesSendBuff = *rowSizesSendBuff_[peerRank];
        auto &mpiColIndicesSendBuff = *entryColIndicesSendBuff_[peerRank];

        // fill the send buffer
        unsigned k = 0;
        for (unsigned i = 0; i < mpiRowIndicesSendBuff.size(); ++i) {
            Index domRowIdx = mpiRowIndicesSendBuff[i];

            for (Index j = 0; j < static_cast<Index>(mpiRowSizesSendBuff[i]); ++j)
            {
                // move to the next column which is in the overlap
                Index domColIdx = mpiColIndicesSendBuff[k];

                // add the values of this column to the send buffer
                mpiSendBuff[k] = (*this)[static_cast<unsigned>(domRowIdx)][static_cast<unsigned>(domColIdx)];
                ++k;
            }
        }

        mpiSendBuff.send(peerRank);
#endif // HAVE_MPI
    }

    void receiveAddEntries_(ProcessRank peerRank)
    {
#if HAVE_MPI
        auto &mpiRecvBuff = *entryValuesRecvBuff_[peerRank];

        auto &mpiRowIndicesRecvBuff = *rowIndicesRecvBuff_[peerRank];
        auto &mpiRowSizesRecvBuff = *rowSizesRecvBuff_[peerRank];
        auto &mpiColIndicesRecvBuff = *entryColIndicesRecvBuff_[peerRank];

        mpiRecvBuff.receive(peerRank);

        // retrieve the values from the receive buffer
        unsigned k = 0;
        for (unsigned i = 0; i < mpiRowIndicesRecvBuff.size(); ++i) {
            Index domRowIdx = mpiRowIndicesRecvBuff[i];
            for (unsigned j = 0; j < mpiRowSizesRecvBuff[i]; ++j, ++k) {
                Index domColIdx = mpiColIndicesRecvBuff[k];

                if (domColIdx < 0)
                    // the matrix for the current process does not know about this DOF
                    continue;

                (*this)[static_cast<unsigned>(domRowIdx)][static_cast<unsigned>(domColIdx)] += mpiRecvBuff[k];
            }
        }
#endif // HAVE_MPI
    }

    void receiveCopyEntries_(int peerRank)
    {
#if HAVE_MPI
        MpiBuffer<block_type> &mpiRecvBuff = *entryValuesRecvBuff_[peerRank];

        MpiBuffer<Index> &mpiRowIndicesRecvBuff = *rowIndicesRecvBuff_[peerRank];
        MpiBuffer<unsigned> &mpiRowSizesRecvBuff = *rowSizesRecvBuff_[peerRank];
        MpiBuffer<Index> &mpiColIndicesRecvBuff = *entryColIndicesRecvBuff_[peerRank];

        mpiRecvBuff.receive(peerRank);

        // retrieve the values from the receive buffer
        unsigned k = 0;
        for (unsigned i = 0; i < mpiRowIndicesRecvBuff.size(); ++i) {
            Index domRowIdx = mpiRowIndicesRecvBuff[i];
            for (unsigned j = 0; j < mpiRowSizesRecvBuff[i]; ++j, ++k) {
                Index domColIdx = mpiColIndicesRecvBuff[k];

                if (domColIdx < 0)
                    // the matrix for the current process does not know about this DOF
                    continue;

                (*this)[static_cast<unsigned>(domRowIdx)][static_cast<unsigned>(domColIdx)] = mpiRecvBuff[k];
            }
        }
#endif // HAVE_MPI
    }

    void globalToDomesticBuff_(MpiBuffer<Index>& idxBuff)
    {
        for (unsigned i = 0; i < idxBuff.size(); ++i)
            idxBuff[i] = overlap_->globalToDomestic(idxBuff[i]);
    }

    int myRank_;
    Entries entries_;
    std::shared_ptr<Overlap> overlap_;

    std::map<ProcessRank, MpiBuffer<unsigned> *> numRowsSendBuff_;
    std::map<ProcessRank, MpiBuffer<unsigned> *> rowSizesSendBuff_;
    std::map<ProcessRank, MpiBuffer<Index> *> rowIndicesSendBuff_;
    std::map<ProcessRank, MpiBuffer<Index> *> entryColIndicesSendBuff_;
    std::map<ProcessRank, MpiBuffer<block_type> *> entryValuesSendBuff_;

    std::map<ProcessRank, MpiBuffer<unsigned> > numRowsRecvBuff_;
    std::map<ProcessRank, MpiBuffer<unsigned> *> rowSizesRecvBuff_;
    std::map<ProcessRank, MpiBuffer<Index> *> rowIndicesRecvBuff_;
    std::map<ProcessRank, MpiBuffer<Index> *> entryColIndicesRecvBuff_;
    std::map<ProcessRank, MpiBuffer<block_type> *> entryValuesRecvBuff_;
};

} // namespace Linear
} // namespace Ewoms

#endif

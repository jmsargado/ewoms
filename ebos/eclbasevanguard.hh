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
 * \copydoc Ewoms::EclBaseVanguard
 */
#ifndef EWOMS_ECL_BASE_VANGUARD_HH
#define EWOMS_ECL_BASE_VANGUARD_HH

#include <ewoms/io/basevanguard.hh>
#include <ewoms/common/propertysystem.hh>
#include <ewoms/common/parametersystem.hh>

#include <opm/parser/eclipse/Parser/Parser.hpp>
#include <opm/parser/eclipse/Parser/ParseContext.hpp>
#include <opm/parser/eclipse/Deck/Deck.hpp>
#include <opm/parser/eclipse/EclipseState/checkDeck.hpp>
#include <opm/parser/eclipse/EclipseState/EclipseState.hpp>
#include <opm/parser/eclipse/EclipseState/Grid/EclipseGrid.hpp>
#include <opm/parser/eclipse/EclipseState/Schedule/Schedule.hpp>
#include <opm/parser/eclipse/EclipseState/SummaryConfig/SummaryConfig.hpp>

#if HAVE_MPI
#include <mpi.h>
#endif // HAVE_MPI

#include <vector>
#include <unordered_set>
#include <array>

namespace Ewoms {
template <class TypeTag>
class EclBaseVanguard;
}

BEGIN_PROPERTIES

NEW_TYPE_TAG(EclBaseVanguard);

// declare the properties required by the for the ecl simulator vanguard
NEW_PROP_TAG(Grid);
NEW_PROP_TAG(EquilGrid);
NEW_PROP_TAG(Scalar);
NEW_PROP_TAG(EclDeckFileName);
NEW_PROP_TAG(OutputDir);
NEW_PROP_TAG(EclOutputInterval);

SET_STRING_PROP(EclBaseVanguard, EclDeckFileName, "");
SET_INT_PROP(EclBaseVanguard, EclOutputInterval, -1); // use the deck-provided value

END_PROPERTIES

namespace Ewoms {

/*!
 * \ingroup EclBlackOilSimulator
 *
 * \brief Helper class for grid instantiation of ECL file-format using problems.
 */
template <class TypeTag>
class EclBaseVanguard : public BaseVanguard<TypeTag>
{
    typedef BaseVanguard<TypeTag> ParentType;
    typedef typename GET_PROP_TYPE(TypeTag, Vanguard) Implementation;
    typedef typename GET_PROP_TYPE(TypeTag, Scalar) Scalar;
    typedef typename GET_PROP_TYPE(TypeTag, Simulator) Simulator;

public:
    typedef typename GET_PROP_TYPE(TypeTag, Grid) Grid;
    typedef typename GET_PROP_TYPE(TypeTag, GridView) GridView;

protected:
    static const int dimension = Grid::dimension;

public:
    /*!
     * \brief Register the common run-time parameters for all ECL simulator vanguards.
     */
    static void registerParameters()
    {
        EWOMS_REGISTER_PARAM(TypeTag, std::string, EclDeckFileName,
                             "The name of the file which contains the ECL deck to be simulated");
        EWOMS_REGISTER_PARAM(TypeTag, int, EclOutputInterval,
                             "The number of report steps that ought to be skipped between two writes of ECL results");
    }

    /*!
     * \brief Returns the canonical path to a deck file.
     *
     * The input can either be the canonical deck file name or the name of the case
     * (i.e., without the .DATA extension)
     */
    static boost::filesystem::path canonicalDeckPath(const std::string& caseName)
    {
        const auto fileExists = [](const boost::filesystem::path& f) -> bool
            {
                if (!boost::filesystem::exists(f))
                    return false;

                if (boost::filesystem::is_regular_file(f))
                    return true;

                return boost::filesystem::is_symlink(f) && boost::filesystem::is_regular_file(boost::filesystem::read_symlink(f));
            };

        auto simcase = boost::filesystem::path(caseName);
        if (fileExists(simcase))
            return simcase;

        for (const auto& ext : { std::string("data"), std::string("DATA") }) {
            if (fileExists(simcase.replace_extension(ext)))
                return simcase;
        }

        throw std::invalid_argument("Cannot find input case '"+caseName+"'");
    }

    /*!
     * \brief Set the Opm::EclipseState and the Opm::Deck object which ought to be used
     *        when the simulator vanguard is instantiated.
     *
     * This is basically an optimization: In cases where the ECL input deck must be
     * examined to decide which simulator ought to be used, this avoids having to parse
     * the input twice. When this method is used, the caller is responsible for lifetime
     * management of these two objects, i.e., they are not allowed to be deleted as long
     * as the simulator vanguard object is alive.
     */
    static void setExternalDeck(Opm::Deck* deck, Opm::EclipseState* eclState, Opm::Schedule* schedule, Opm::SummaryConfig* summaryConfig)
    {
        externalDeck_ = deck;
        externalEclState_ = eclState;
        externalSchedule_ = schedule;
        externalSummaryConfig_ = summaryConfig;
    }

    /*!
     * \brief Create the grid for problem data files which use the ECL file format.
     *
     * This is the file format used by the commercial ECLiPSE simulator. Usually it uses
     * a cornerpoint description of the grid.
     */
    EclBaseVanguard(Simulator& simulator)
        : ParentType(simulator)
    {
        int myRank = 0;
#if HAVE_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
#endif

        std::string fileName = EWOMS_GET_PARAM(TypeTag, std::string, EclDeckFileName);

        if (fileName == "")
            throw std::runtime_error("No input deck file has been specified as a command line argument,"
                                     " or via '--ecl-deck-file-name=CASE.DATA'");

        fileName = canonicalDeckPath(fileName).string();

        // compute the base name of the input file name
        const char directorySeparator = '/';
        long int i;
        for (i = fileName.size(); i >= 0; -- i)
            if (fileName[i] == directorySeparator)
                break;
        std::string baseName = fileName.substr(i + 1, fileName.size());

        // remove the extension from the input file
        for (i = baseName.size(); i >= 0; -- i)
            if (baseName[i] == '.')
                break;
        std::string rawCaseName;
        if (i < 0)
            rawCaseName = baseName;
        else
            rawCaseName = baseName.substr(0, i);

        // transform the result to ALL_UPPERCASE
        caseName_ = rawCaseName;
        std::transform(caseName_.begin(), caseName_.end(), caseName_.begin(), ::toupper);

        if (!externalDeck_) {
            if (myRank == 0)
                std::cout << "Reading the deck file '" << fileName << "'" << std::endl;

            Opm::Parser parser;
            typedef std::pair<std::string, Opm::InputError::Action> ParseModePair;
            typedef std::vector<ParseModePair> ParseModePairs;

            ParseModePairs tmp;
            tmp.emplace_back(Opm::ParseContext::PARSE_RANDOM_SLASH, Opm::InputError::IGNORE);
            tmp.emplace_back(Opm::ParseContext::PARSE_MISSING_DIMS_KEYWORD, Opm::InputError::WARN);
            tmp.emplace_back(Opm::ParseContext::SUMMARY_UNKNOWN_WELL, Opm::InputError::WARN);
            tmp.emplace_back(Opm::ParseContext::SUMMARY_UNKNOWN_GROUP, Opm::InputError::WARN);
            Opm::ParseContext parseContext(tmp);

            internalDeck_.reset(new Opm::Deck(parser.parseFile(fileName , parseContext)));
            internalEclState_.reset(new Opm::EclipseState(*internalDeck_, parseContext));
            {
                const auto& grid = internalEclState_->getInputGrid();
                const Opm::TableManager table ( *internalDeck_ );
                const Opm::Eclipse3DProperties eclipseProperties (*internalDeck_  , table, grid);
                internalSchedule_.reset(new Opm::Schedule(*internalDeck_, grid, eclipseProperties, Opm::Phases(true, true, true), parseContext ));
                internalSummaryConfig_.reset(new Opm::SummaryConfig(*internalDeck_, *internalSchedule_, table,  parseContext));
            }


            deck_ = &(*internalDeck_);
            eclState_ = &(*internalEclState_);
            summaryConfig_ = &(*internalSummaryConfig_);
            schedule_ = &(*internalSchedule_);
        }
        else {
            assert(externalDeck_);
            assert(externalEclState_);
            assert(externalSchedule_);
            assert(externalSummaryConfig_);

            deck_ = externalDeck_;
            eclState_ = externalEclState_;
            schedule_ = externalSchedule_;
            summaryConfig_ = externalSummaryConfig_;
        }

        // Possibly override IOConfig setting for how often RESTART files should get
        // written to disk (every N report step)
        int outputInterval = EWOMS_GET_PARAM(TypeTag, int, EclOutputInterval);
        if (outputInterval >= 0)
            eclState_->getRestartConfig().overrideRestartWriteInterval(outputInterval);

        asImp_().createGrids_();
        asImp_().filterConnections_();
        asImp_().updateOutputDir_();
        asImp_().finalizeInit_();
    }

    /*!
     * \brief Return a pointer to the parsed ECL deck
     */
    const Opm::Deck& deck() const
    { return *deck_; }

    Opm::Deck& deck()
    { return *deck_; }

    /*!
     * \brief Return a pointer to the internalized ECL deck
     */
    const Opm::EclipseState& eclState() const
    { return *eclState_; }

    Opm::EclipseState& eclState()
    { return *eclState_; }

    const Opm::Schedule& schedule() const {
        return *schedule_;
    }

    Opm::Schedule& schedule() {
        return *schedule_;
    }

    const Opm::SummaryConfig& summaryConfig() const {
        return *summaryConfig_;
    }
    /*!
     * \brief Returns the name of the case.
     *
     * i.e., the all-uppercase version of the file name from which the
     * deck is loaded with the ".DATA" suffix removed.
     */
    const std::string& caseName() const
    { return caseName_; }

    /*!
     * \brief Returns the number of logically Cartesian cells in each direction
     */
    const std::array<int, dimension>& cartesianDimensions() const
    { return asImp_().cartesianIndexMapper().cartesianDimensions(); }

    /*!
     * \brief Returns the overall number of cells of the logically Cartesian grid
     */
    int cartesianSize() const
    { return asImp_().cartesianIndexMapper().cartesianSize(); }

    /*!
     * \brief Returns the overall number of cells of the logically EquilCartesian grid
     */
    int equilCartesianSize() const
    { return asImp_().equilCartesianIndexMapper().cartesianSize(); }

    /*!
     * \brief Returns the Cartesian cell id for identifaction with ECL data
     */
    unsigned cartesianIndex(unsigned compressedCellIdx) const
    { return asImp_().cartesianIndexMapper().cartesianIndex(compressedCellIdx); }

    /*!
     * \brief Return the index of the cells in the logical Cartesian grid
     */
    unsigned cartesianIndex(const std::array<int,dimension>& coords) const
    {
        unsigned cartIndex = coords[0];
        int factor = cartesianDimensions()[0];
        for (unsigned i = 1; i < dimension; ++i) {
            cartIndex += coords[i]*factor;
            factor *= cartesianDimensions()[i];
        }

        return cartIndex;
    }

    /*!
     * \brief Extract Cartesian index triplet (i,j,k) of an active cell.
     *
     * \param [in] cellIdx Active cell index.
     * \param [out] ijk Cartesian index triplet
     */
    void cartesianCoordinate(unsigned cellIdx, std::array<int,3>& ijk) const
    { return asImp_().cartesianIndexMapper().cartesianCoordinate(cellIdx, ijk); }

    /*!
     * \brief Returns the Cartesian cell id given an element index for the grid used for equilibration
     */
    unsigned equilCartesianIndex(unsigned compressedEquilCellIdx) const
    { return asImp_().equilCartesianIndexMapper().cartesianIndex(compressedEquilCellIdx); }

    /*!
     * \brief Extract Cartesian index triplet (i,j,k) of an active cell of the grid used for EQUIL.
     *
     * \param [in] cellIdx Active cell index.
     * \param [out] ijk Cartesian index triplet
     */
    void equilCartesianCoordinate(unsigned cellIdx, std::array<int,3>& ijk) const
    { return asImp_().equilCartesianIndexMapper().cartesianCoordinate(cellIdx, ijk); }

    /*!
     * \brief Return the names of the wells which do not penetrate any cells on the local
     *        process.
     *
     * This is a kludge around the fact that for distributed grids, not all wells are
     * seen by all proccesses.
     */
    std::unordered_set<std::string> defunctWellNames() const
    { return std::unordered_set<std::string>(); }

private:
    void updateOutputDir_()
    {
        // update the location for output
        std::string outputDir = EWOMS_GET_PARAM(TypeTag, std::string, OutputDir);
        auto& ioConfig = eclState_->getIOConfig();
        if (outputDir == "")
            // If no output directory parameter is specified, use the output directory
            // which Opm::IOConfig thinks that should be used. Normally this is the
            // directory in which the input files are located.
            outputDir = ioConfig.getOutputDir();

        // ensure that the output directory exists and that it is a directory
        if (!boost::filesystem::is_directory(outputDir)) {
            try {
                boost::filesystem::create_directories(outputDir);
            }
            catch (...) {
                 throw std::runtime_error("Creation of output directory '"+outputDir+"' failed\n");
            }
        }

        // specify the directory output. This is not a very nice mechanism because
        // the eclState is supposed to be immutable here, IMO.
        ioConfig.setOutputDir(outputDir);
    }

    Implementation& asImp_()
    { return *static_cast<Implementation*>(this); }

    const Implementation& asImp_() const
    { return *static_cast<const Implementation*>(this); }

    std::string caseName_;

    static Opm::Deck* externalDeck_;
    static Opm::EclipseState* externalEclState_;
    static Opm::Schedule* externalSchedule_;
    static Opm::SummaryConfig* externalSummaryConfig_;
    std::unique_ptr<Opm::Deck> internalDeck_;
    std::unique_ptr<Opm::EclipseState> internalEclState_;
    std::unique_ptr<Opm::Schedule> internalSchedule_;
    std::unique_ptr<Opm::SummaryConfig> internalSummaryConfig_;

    // these two attributes point either to the internal or to the external version of the
    // Deck and EclipsState objects.
    Opm::Deck* deck_;
    Opm::EclipseState* eclState_;
    Opm::Schedule* schedule_;
    Opm::SummaryConfig* summaryConfig_;
};

template <class TypeTag>
Opm::Deck* EclBaseVanguard<TypeTag>::externalDeck_ = nullptr;

template <class TypeTag>
Opm::EclipseState* EclBaseVanguard<TypeTag>::externalEclState_;

template <class TypeTag>
Opm::Schedule* EclBaseVanguard<TypeTag>::externalSchedule_ = nullptr;

template <class TypeTag>
Opm::SummaryConfig* EclBaseVanguard<TypeTag>::externalSummaryConfig_ = nullptr;


} // namespace Ewoms

#endif

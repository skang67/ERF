#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_WriteEBSurface.H>

#include <ERF_EBIFTerrain.H>
#include <ERF_ProbCommon.H>

#include <ERF.H>
#include <ERF_EB.H>

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EB2.H>
#include <AMReX_EBToPVD.H>
#include <AMReX_iMultiFab.H>

using namespace amrex;

eb_::~eb_()
{
    // if (m_factory) { m_factory.reset(nullptr); }
}

eb_::eb_ ( )
    : m_has_eb(0),
      m_support_level(EBSupport::full),
      m_write_eb_surface(0)
{ }

void
eb_::make_all_factories ([[maybe_unused]] int level,
                        Geometry            const& a_geom,
                        BoxArray            const& ba,
                        DistributionMapping const& dm,
                        EB2::Level const& a_eb_level)
{
    Print() << "making EB factory\n";
    m_factory = std::make_unique<EBFArrayBoxFactory>(a_eb_level, a_geom, ba, dm,
        Vector<int>{nghost_basic(), nghost_volume(), nghost_full()}, m_support_level);

    // Correct cell connectivity
    eb_::set_connection_flags();

    { int const idim(0);
        Print() << "making EB staggered u-factory\n";
        //m_u_factory.set_verbose();
        m_u_factory.define(level, idim, a_geom, ba, dm,
            Vector<int>{nghost_basic(), nghost_volume(), nghost_full()},
            m_factory.get());
    }

    { int const idim(1);
        Print() << "making EB staggered v-factory\n";
        //m_v_factory.set_verbose();
        m_v_factory.define(level, idim, a_geom, ba, dm,
            Vector<int>{nghost_basic(), nghost_volume(), nghost_full()},
            m_factory.get());
    }

    { int const idim(2);
        Print() << "making EB staggered w-factory\n";
        //m_w_factory.set_verbose();
        m_w_factory.define(level, idim, a_geom, ba, dm,
            Vector<int>{nghost_basic(), nghost_volume(), nghost_full()},
            m_factory.get());
    }
    Print() << "\nDone making EB factory at level = " << level << ".\n\n";

    // Compute per-column first uncovered w-face index (m_k_terrain).
    // For terrain-only EB, covered z-faces are contiguous from the bottom.
    // m_k_terrain(i,j,0) = first k where flag_w is NOT covered.
    // If the entire column is covered, sentinel = domain_hi_z + 2.
    {
        // Build a 2D (x,y) BoxArray from the cell-centered BA collapsed in z
        BoxArray ba2d = ba;
        ba2d.enclosedCells();  // ensure cell-centered
        BoxList bl2d;
        for (int idx = 0; idx < ba2d.size(); ++idx) {
            Box bx2d = ba2d[idx];
            bx2d.setRange(2, 0);  // flatten to k=0
            bl2d.push_back(bx2d);
        }
        BoxArray ba_flat(std::move(bl2d));

        m_k_terrain.define(ba_flat, dm, 1, 0);

        const int domhi_z = a_geom.Domain().bigEnd(2);
        const int sentinel = domhi_z + 2;

        for (MFIter mfi(m_k_terrain, false); mfi.isValid(); ++mfi) {
            const Box& bx2 = mfi.validbox();
            const auto& k_terr = m_k_terrain.array(mfi);
            const auto& flag_w = m_w_factory.getMultiEBCellFlagFab()[mfi].const_array();

            // Get the full z-range from the w-factory's box for this MFIter
            const Box& wbox = m_w_factory.getMultiEBCellFlagFab()[mfi].box();
            const int klo = wbox.smallEnd(2);
            const int khi = wbox.bigEnd(2);

            ParallelFor(bx2, [=] AMREX_GPU_DEVICE (int i, int j, int) noexcept
            {
                int k_first = sentinel;
                for (int k = klo; k <= khi; ++k) {
                    if (!flag_w(i,j,k).isCovered()) {
                        k_first = k;
                        break;
                    }
                }
                k_terr(i,j,0) = k_first;
            });
        }
        // Print() << "Computed k_terrain (per-column first uncovered w-face)\n";
    }
}

void
eb_::make_cc_factory ([[maybe_unused]] int level,
                        Geometry            const& a_geom,
                        BoxArray            const& ba,
                        DistributionMapping const& dm,
                        EB2::Level const& a_eb_level)
{
    Print() << "making EB factory\n";
    m_factory = std::make_unique<EBFArrayBoxFactory>(a_eb_level, a_geom, ba, dm,
        Vector<int>{nghost_basic(), nghost_volume(), nghost_full()}, m_support_level);

    Print() << "\nDone making EB factory at level " << level << ".\n\n";
}

/*
Reset cell flags to disconnect cells with zero volume fraction,
via non-const reference from EBFArrayBoxFactory.
*/
void
eb_::set_connection_flags ()
{
    // Get non-const reference to EBCellFlagFab FabArray
    FabArray<EBCellFlagFab>& cellflag = getNonConstEBCellFlags(*m_factory);

    const MultiFab& volfrac = m_factory->getVolFrac();

    for (MFIter mfi(cellflag, false); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const Box gbx = amrex::grow(bx, cellflag.nGrow()-1); // Leave one cell layer

        Array4<EBCellFlag> const& flag = cellflag.array(mfi);
        Array4<Real const> const& vfrac = volfrac.const_array(mfi);

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            for(int kk(-1); kk<=1; kk++) {
            for(int jj(-1); jj<=1; jj++) {
            for(int ii(-1); ii<=1; ii++)
            {
                if (vfrac(i+ii,j+jj,k+kk) == zero) {
                    flag(i,j,k).setDisconnected(ii,jj,kk);
                }
            }}}
        });

        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (vfrac(i,j,k)==zero) {
                flag(i,j,k).setCovered();
            }
        });

    }
}

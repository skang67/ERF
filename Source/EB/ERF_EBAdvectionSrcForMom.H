/**
 * \file ERF_EBAdvectionSrcForMom.H
 * \brief Defines EB momentum-advection flux builders for higher-order stencils.
 */
#include <ERF_IndexDefines.H>
#include <ERF_Interpolation.H>

namespace erf_advection_detail {

/**
 * \brief Return whether the complete directional interpolation footprint is regular.
 *
 * For a flux at index i, a positive-flow scan reads [i-N,i-1] and a
 * negative-flow scan reads [i,i+N-1].  Growing the complete flux box by N on
 * the low side and N-1 on the high side therefore covers both signs.
 */
[[nodiscard]] inline bool
momentum_stencil_is_regular (const amrex::EBCellFlagFab& flag_fab,
                             const amrex::Box& flux_box,
                             const int dir,
                             const int stencil_size)
{
    amrex::Box stencil_box = amrex::convert(flux_box, flag_fab.box().ixType());
    stencil_box.growLo(dir, stencil_size);
    stencil_box.growHi(dir, stencil_size - 1);

    return flag_fab.box().contains(stencil_box) &&
           flag_fab.getType(stencil_box) == amrex::FabType::regular;
}

[[nodiscard]] inline amrex::Box
paired_flux_box (const amrex::Box& flux_box, int dir)
{
    amrex::Box result = flux_box;
    result.setBig(dir, flux_box.smallEnd(dir) + (flux_box.length(dir) - 1) / 2);
    return result;
}

template<int Dir, typename InterpType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
regular_interpolate (const InterpType& high_order,
                     const CENTERED2& centered,
                     int i, int j, int k,
                     amrex::Real mass_flux,
                     amrex::Real& value) noexcept
{
    if (mass_flux > zero || mass_flux < zero) {
        if constexpr (Dir == 0) {
            high_order.InterpolateInX(i,j,k,0,value,mass_flux);
        } else if constexpr (Dir == 1) {
            high_order.InterpolateInY(i,j,k,0,value,mass_flux);
        } else {
            high_order.InterpolateInZ(i,j,k,0,value,mass_flux);
        }
    } else {
        if constexpr (Dir == 0) {
            centered.InterpolateInX(i,j,k,0,value,mass_flux);
        } else if constexpr (Dir == 1) {
            centered.InterpolateInY(i,j,k,0,value,mass_flux);
        } else {
            centered.InterpolateInZ(i,j,k,0,value,mass_flux);
        }
    }
}

template<int Dir, typename InterpType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
regular_interpolate_pair (const InterpType& high_order,
                          const CENTERED2& centered,
                          int i, int j, int k,
                          amrex::Real mass_flux_lo,
                          amrex::Real mass_flux_hi,
                          amrex::Real& value_lo,
                          amrex::Real& value_hi) noexcept
{
    bool const high_order_lo = mass_flux_lo > zero || mass_flux_lo < zero;
    bool const high_order_hi = mass_flux_hi > zero || mass_flux_hi < zero;

    if (high_order_lo && high_order_hi) {
        high_order.template InterpolatePair<Dir>(i, j, k, 0, value_lo, value_hi,
                                                 mass_flux_lo, mass_flux_hi);
    } else {
        regular_interpolate<Dir>(high_order, centered, i, j, k, mass_flux_lo, value_lo);
        if constexpr (Dir == 0) {
            regular_interpolate<Dir>(high_order, centered, i+1, j, k, mass_flux_hi, value_hi);
        } else if constexpr (Dir == 1) {
            regular_interpolate<Dir>(high_order, centered, i, j+1, k, mass_flux_hi, value_hi);
        } else {
            regular_interpolate<Dir>(high_order, centered, i, j, k+1, mass_flux_hi, value_hi);
        }
    }
}

template<int StencilSize>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int
regular_w_vertical_cells (int k, int lo_z_face, int hi_z_face,
                          amrex::Real mass_flux) noexcept
{
    int cells = (mass_flux > zero || mass_flux < zero) ? StencilSize : 0;
    if (k == hi_z_face || k == lo_z_face+1) {
        cells = amrex::min(cells, 1);
    } else if (k == hi_z_face-1 || k == lo_z_face+2) {
        cells = amrex::min(cells, 2);
    }
    return cells;
}

template<int StencilSize, typename InterpType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
regular_w_vertical_interpolate (const InterpType& high_order,
                                const CENTERED2& centered,
                                const UPWIND3& upwind3,
                                int i, int j, int k, int cells,
                                amrex::Real mass_flux,
                                amrex::Real& value) noexcept
{
    if (cells == StencilSize) {
        high_order.InterpolateInZ(i,j,k,0,value,mass_flux);
    } else if (cells <= 1) {
        centered.InterpolateInZ(i,j,k,0,value,mass_flux);
    } else if (cells == 2) {
        upwind3.InterpolateInZ(i,j,k,0,value,mass_flux);
    }
}

template<int StencilSize, typename InterpType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void
regular_w_vertical_interpolate_pair (const InterpType& high_order,
                                     const CENTERED2& centered,
                                     const UPWIND3& upwind3,
                                     int i, int j, int k,
                                     int cells_lo, int cells_hi,
                                     amrex::Real mass_flux_lo,
                                     amrex::Real mass_flux_hi,
                                     amrex::Real& value_lo,
                                     amrex::Real& value_hi) noexcept
{
    if (cells_lo == StencilSize && cells_hi == StencilSize) {
        high_order.template InterpolatePair<2>(i, j, k, 0, value_lo, value_hi,
                                               mass_flux_lo, mass_flux_hi);
    } else if (cells_lo <= 1 && cells_hi <= 1) {
        centered.template InterpolatePair<2>(i, j, k, 0, value_lo, value_hi,
                                             mass_flux_lo, mass_flux_hi);
    } else if (cells_lo == 2 && cells_hi == 2) {
        upwind3.template InterpolatePair<2>(i, j, k, 0, value_lo, value_hi,
                                           mass_flux_lo, mass_flux_hi);
    } else {
        regular_w_vertical_interpolate<StencilSize>(
            high_order, centered, upwind3, i, j, k, cells_lo, mass_flux_lo, value_lo);
        regular_w_vertical_interpolate<StencilSize>(
            high_order, centered, upwind3, i, j, k+1, cells_hi, mass_flux_hi, value_hi);
    }
}

} // namespace erf_advection_detail

/**
 * \brief Build EB momentum fluxes with selected horizontal and vertical interpolation.
 *
 * The wrapper handles the x-, y-, and z-momentum grids separately and reduces
 * interpolation order near covered EB cells and vertical domain boundaries.
 *
 * \tparam InterpType_H Horizontal interpolation functor.
 * \tparam InterpType_V Vertical interpolation functor.
 * \tparam WallInterpType Wall interpolation functor retained for interface symmetry.
 */
template<typename InterpType_H, typename InterpType_V, typename WallInterpType>
void
EBAdvectionSrcForMomWrapper (const amrex::Vector<amrex::Box>& bxx_grown,
                            const amrex::Vector<amrex::Box>& bxy_grown,
                            const amrex::Vector<amrex::Box>& bxz_grown,
                            const amrex::Array4<const amrex::Real>& rho_u,
                            const amrex::Array4<const amrex::Real>& rho_v,
                            const amrex::Array4<const amrex::Real>& rho_w,
                            const amrex::Array4<const amrex::Real>& u,
                            const amrex::Array4<const amrex::Real>& v,
                            const amrex::Array4<const amrex::Real>& w,
                            const amrex::FabType u_type,
                            const amrex::FabType v_type,
                            const amrex::FabType w_type,
                            const amrex::EBCellFlagFab& u_flag_fab,
                            const amrex::EBCellFlagFab& v_flag_fab,
                            const amrex::EBCellFlagFab& w_flag_fab,
                            const amrex::Array4<const amrex::EBCellFlag>& u_cflag,
                            const amrex::Array4<const amrex::Real>& u_afrac_x,
                            const amrex::Array4<const amrex::Real>& u_afrac_y,
                            const amrex::Array4<const amrex::Real>& u_afrac_z,
                            const amrex::Array4<const amrex::EBCellFlag>& v_cflag,
                            const amrex::Array4<const amrex::Real>& v_afrac_x,
                            const amrex::Array4<const amrex::Real>& v_afrac_y,
                            const amrex::Array4<const amrex::Real>& v_afrac_z,
                            const amrex::Array4<const amrex::EBCellFlag>& w_cflag,
                            const amrex::Array4<const amrex::Real>& w_afrac_x,
                            const amrex::Array4<const amrex::Real>& w_afrac_y,
                            const amrex::Array4<const amrex::Real>& w_afrac_z,
                            const amrex::Array4<const amrex::Real>& mf_ux_inv,
                            const amrex::Array4<const amrex::Real>& mf_vx_inv,
                            const amrex::Array4<const amrex::Real>& mf_uy_inv,
                            const amrex::Array4<const amrex::Real>& mf_vy_inv,
                            const amrex::Real upw_frac_h,
                            const amrex::Real upw_frac_v,
                            const AdvType /*vert_adv_type*/,
                                  amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_u_arr,
                                  amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_v_arr,
                                  amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_w_arr,
                            const int lo_z_face, const int hi_z_face)
{
    // Instantiate the appropriate structs
    InterpType_H interp_u_horz(u, upw_frac_h); InterpType_V interp_u_vert(u, upw_frac_v); // X-MOM
    InterpType_H interp_v_horz(v, upw_frac_h); InterpType_V interp_v_vert(v, upw_frac_v); // Y-MOM
    InterpType_H interp_w_horz(w, upw_frac_h); InterpType_V interp_w_vert(w, upw_frac_v); // Z-MOM
    // WallInterpType interp_w_wall(w, upw_frac_v); // Z-MOM @ wall

    const int ncells_u_horz = interp_u_horz.GetUpwindCellNumber();
    const int ncells_u_vert = interp_u_vert.GetUpwindCellNumber();
    const int ncells_v_horz = interp_v_horz.GetUpwindCellNumber();
    const int ncells_v_vert = interp_v_vert.GetUpwindCellNumber();
    const int ncells_w_horz = interp_w_horz.GetUpwindCellNumber();
    const int ncells_w_vert = interp_w_vert.GetUpwindCellNumber();

    static constexpr int horizontal_stencil_size = InterpType_H::GetUpwindCellNumber();
    static constexpr int vertical_stencil_size = InterpType_V::GetUpwindCellNumber();

    amrex::GpuArray<bool, AMREX_SPACEDIM> u_stencil_is_regular{false, false, false};
    amrex::GpuArray<bool, AMREX_SPACEDIM> v_stencil_is_regular{false, false, false};
    amrex::GpuArray<bool, AMREX_SPACEDIM> w_stencil_is_regular{false, false, false};

    if (u_type == amrex::FabType::regular) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int stencil_size = (dir < 2) ? horizontal_stencil_size
                                               : vertical_stencil_size;
            u_stencil_is_regular[dir] = erf_advection_detail::momentum_stencil_is_regular(
                u_flag_fab, bxx_grown[dir], dir, stencil_size);
        }
    }
    if (v_type == amrex::FabType::regular) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int stencil_size = (dir < 2) ? horizontal_stencil_size
                                               : vertical_stencil_size;
            v_stencil_is_regular[dir] = erf_advection_detail::momentum_stencil_is_regular(
                v_flag_fab, bxy_grown[dir], dir, stencil_size);
        }
    }
    if (w_type == amrex::FabType::regular) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            const int stencil_size = (dir < 2) ? horizontal_stencil_size
                                               : vertical_stencil_size;
            w_stencil_is_regular[dir] = erf_advection_detail::momentum_stencil_is_regular(
                w_flag_fab, bxz_grown[dir], dir, stencil_size);
        }
    }

    // Instantiate structs for lower-order vert/horiz interp (commonly using upw_frac_h)
    CENTERED2 interp_u_CEN2(u, 0);
    CENTERED2 interp_v_CEN2(v, 0);
    CENTERED2 interp_w_CEN2(w, 0);

    UPWIND3   interp_u_UPW3(u, upw_frac_h);
    UPWIND3   interp_v_UPW3(v, upw_frac_h);
    UPWIND3   interp_w_UPW3(w, upw_frac_h);

    bool u_is_regular = (u_type == amrex::FabType::regular);
    bool v_is_regular = (v_type == amrex::FabType::regular);
    bool w_is_regular = (w_type == amrex::FabType::regular);

    bool u_is_singlevalued = (u_type == amrex::FabType::singlevalued);
    bool v_is_singlevalued = (v_type == amrex::FabType::singlevalued);
    bool w_is_singlevalued = (w_type == amrex::FabType::singlevalued);

    const bool u_x_stencil_is_regular = u_stencil_is_regular[0];
    const bool u_y_stencil_is_regular = u_stencil_is_regular[1];
    const bool u_z_stencil_is_regular = u_stencil_is_regular[2];
    const bool v_x_stencil_is_regular = v_stencil_is_regular[0];
    const bool v_y_stencil_is_regular = v_stencil_is_regular[1];
    const bool v_z_stencil_is_regular = v_stencil_is_regular[2];
    const bool w_x_stencil_is_regular = w_stencil_is_regular[0];
    const bool w_y_stencil_is_regular = w_stencil_is_regular[1];
    const bool w_z_stencil_is_regular = w_stencil_is_regular[2];

    const bool u_has_regular_stencils = u_x_stencil_is_regular &&
                                        u_y_stencil_is_regular &&
                                        u_z_stencil_is_regular;
    const bool v_has_regular_stencils = v_x_stencil_is_regular &&
                                        v_y_stencil_is_regular &&
                                        v_z_stencil_is_regular;
    const bool w_has_regular_stencils = w_x_stencil_is_regular &&
                                        w_y_stencil_is_regular &&
                                        w_z_stencil_is_regular;

    // ---------------------------------------------------- //
    // x-momentum
    // ---------------------------------------------------- //

    // Use a distinct kernel type so proven-regular launches do not capture or
    // compile EB area fractions, flag scans, or unreachable order fallbacks.
    if (u_has_regular_stencils) {
        amrex::Box const u_x_pair_box = erf_advection_detail::paired_flux_box(bxx_grown[0], 0);
        amrex::Box const u_y_pair_box = erf_advection_detail::paired_flux_box(bxx_grown[1], 1);
        amrex::Box const u_z_pair_box = erf_advection_detail::paired_flux_box(bxx_grown[2], 2);
        int const u_x_lo = bxx_grown[0].smallEnd(0);
        int const u_y_lo = bxx_grown[1].smallEnd(1);
        int const u_z_lo = bxx_grown[2].smallEnd(2);
        int const u_x_hi = bxx_grown[0].bigEnd(0);
        int const u_y_hi = bxx_grown[1].bigEnd(1);
        int const u_z_hi = bxx_grown[2].bigEnd(2);

        amrex::ParallelFor(u_x_pair_box, u_y_pair_box, u_z_pair_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const ii = u_x_lo + 2 * (i - u_x_lo);
            amrex::Real const rho_um1 = rho_u(ii-1,j,k);
            amrex::Real const rho_u0 = rho_u(ii,j,k);
            amrex::Real const mf_um1 = mf_ux_inv(ii-1,j,0);
            amrex::Real const mf_u0 = mf_ux_inv(ii,j,0);
            amrex::Real const rho_u_avg0 = myhalf * (rho_um1 * mf_um1 + rho_u0 * mf_u0);
            amrex::Real interpx0 = zero;

            if (ii < u_x_hi) {
                amrex::Real const rho_up1 = rho_u(ii+1,j,k);
                amrex::Real const mf_up1 = mf_ux_inv(ii+1,j,0);
                amrex::Real const rho_u_avg1 = myhalf * (rho_u0 * mf_u0 + rho_up1 * mf_up1);
                amrex::Real interpx1 = zero;
                erf_advection_detail::regular_interpolate_pair<0>(
                    interp_u_horz, interp_u_CEN2, ii, j, k,
                    rho_u_avg0, rho_u_avg1, interpx0, interpx1);
                flx_u_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
                flx_u_arr[0](ii+1,j,k) = rho_u_avg1 * interpx1;
            } else {
                erf_advection_detail::regular_interpolate<0>(
                    interp_u_horz, interp_u_CEN2, ii, j, k, rho_u_avg0, interpx0);
                flx_u_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const jj = u_y_lo + 2 * (j - u_y_lo);
            amrex::Real const rho_v_avg0 = myhalf * (
                rho_v(i,jj,k) * mf_vx_inv(i,jj,0) +
                rho_v(i-1,jj,k) * mf_vx_inv(i-1,jj,0));
            amrex::Real interpy0 = zero;

            if (jj < u_y_hi) {
                amrex::Real const rho_v_avg1 = myhalf * (
                    rho_v(i,jj+1,k) * mf_vx_inv(i,jj+1,0) +
                    rho_v(i-1,jj+1,k) * mf_vx_inv(i-1,jj+1,0));
                amrex::Real interpy1 = zero;
                erf_advection_detail::regular_interpolate_pair<1>(
                    interp_u_horz, interp_u_CEN2, i, jj, k,
                    rho_v_avg0, rho_v_avg1, interpy0, interpy1);
                flx_u_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
                flx_u_arr[1](i,jj+1,k) = rho_v_avg1 * interpy1;
            } else {
                erf_advection_detail::regular_interpolate<1>(
                    interp_u_horz, interp_u_CEN2, i, jj, k, rho_v_avg0, interpy0);
                flx_u_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const kk = u_z_lo + 2 * (k - u_z_lo);
            amrex::Real const rho_w_avg0 = myhalf * (rho_w(i,j,kk) + rho_w(i-1,j,kk));
            amrex::Real interpz0 = zero;

            if (kk < u_z_hi) {
                amrex::Real const rho_w_avg1 = myhalf * (rho_w(i,j,kk+1) + rho_w(i-1,j,kk+1));
                amrex::Real interpz1 = zero;
                erf_advection_detail::regular_interpolate_pair<2>(
                    interp_u_vert, interp_u_CEN2, i, j, kk,
                    rho_w_avg0, rho_w_avg1, interpz0, interpz1);
                flx_u_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
                flx_u_arr[2](i,j,kk+1) = rho_w_avg1 * interpz1;
            } else {
                erf_advection_detail::regular_interpolate<2>(
                    interp_u_vert, interp_u_CEN2, i, j, kk, rho_w_avg0, interpz0);
                flx_u_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
            }
        });
    } else {
        amrex::ParallelFor(bxx_grown[0], bxx_grown[1], bxx_grown[2],
    // x-momentum: x-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = u_is_regular || (u_is_singlevalued && u_afrac_x(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_u_avg = myhalf * (rho_u(i-1,j,k) * mf_ux_inv(i-1,j,0) + rho_u(i,j,k) * mf_ux_inv(i,j,0));

            int icell = 0;

            if (u_x_stencil_is_regular)
            {
                if (rho_u_avg > zero || rho_u_avg < zero) {
                    icell = ncells_u_horz;
                }
            }
            else if (rho_u_avg>zero)
            {
                for (int ii=0; ii<ncells_u_horz; ++ii) {
                    if (u_cflag(i-ii-1,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }
            else if (rho_u_avg<zero)
            {
                for (int ii=0; ii<ncells_u_horz; ++ii) {
                    if (u_cflag(i+ii,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==ncells_u_horz) {
                interp_u_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                if (icell<=1) {
                    interp_u_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                } else if (icell==2) {
                    interp_u_UPW3.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                }
            }
            flx_u_arr[0](i,j,k) = rho_u_avg * interpx;

        } else {

            flx_u_arr[0](i,j,k) = zero;
        }
    },
    // x-momentum: y-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = u_is_regular || (u_is_singlevalued && u_afrac_y(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) * mf_vx_inv(i,j,0) + rho_v(i-1,j,k) * mf_vx_inv(i-1,j,0));

            int jcell = 0;

            if (u_y_stencil_is_regular)
            {
                if (rho_v_avg > zero || rho_v_avg < zero) {
                    jcell = ncells_u_horz;
                }
            }
            else if (rho_v_avg>zero)
            {
                for (int jj=0; jj<ncells_u_horz; ++jj) {
                    if (u_cflag(i,j-jj-1,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }
            else if (rho_v_avg<zero)
            {
                for (int jj=0; jj<ncells_u_horz; ++jj) {
                    if (u_cflag(i,j+jj,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==ncells_u_horz) {
                interp_u_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                if (jcell<=1) {
                    interp_u_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                } else if (jcell==2) {
                    interp_u_UPW3.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                }
            }
            flx_u_arr[1](i,j,k) = rho_v_avg * interpy;

        } else {

            flx_u_arr[1](i,j,k) = zero;
        }
    },
    // x-momentum: z-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = u_is_regular || (u_is_singlevalued && u_afrac_z(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i-1,j,k));

            int kcell = 0;

            if (u_z_stencil_is_regular)
            {
                if (rho_w_avg > zero || rho_w_avg < zero) {
                    kcell = ncells_u_vert;
                }
            }
            else if (rho_w_avg>zero)
            {
                for (int kk=0; kk<ncells_u_vert; ++kk) {
                    if (u_cflag(i,j,k-kk-1).isCovered()) {
                        break;
                    }
                    kcell++;
                }
            }
            else if (rho_w_avg<zero)
            {
                for (int kk=0; kk<ncells_u_vert; ++kk) {
                    if (u_cflag(i,j,k+kk).isCovered()) {
                        break;
                    }
                    kcell++;
                }
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpz = zero;
            if (kcell==ncells_u_vert) {
                interp_u_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            } else {
                if (kcell<=1) {
                    interp_u_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                } else if (kcell==2) {
                    interp_u_UPW3.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                }
            }
            flx_u_arr[2](i,j,k) = rho_w_avg * interpz;

        } else {

            flx_u_arr[2](i,j,k) = zero;
        }
        });
    }

    // ---------------------------------------------------- //
    // y-momentum
    // ---------------------------------------------------- //

    if (v_has_regular_stencils) {
        amrex::Box const v_x_pair_box = erf_advection_detail::paired_flux_box(bxy_grown[0], 0);
        amrex::Box const v_y_pair_box = erf_advection_detail::paired_flux_box(bxy_grown[1], 1);
        amrex::Box const v_z_pair_box = erf_advection_detail::paired_flux_box(bxy_grown[2], 2);
        int const v_x_lo = bxy_grown[0].smallEnd(0);
        int const v_y_lo = bxy_grown[1].smallEnd(1);
        int const v_z_lo = bxy_grown[2].smallEnd(2);
        int const v_x_hi = bxy_grown[0].bigEnd(0);
        int const v_y_hi = bxy_grown[1].bigEnd(1);
        int const v_z_hi = bxy_grown[2].bigEnd(2);

        amrex::ParallelFor(v_x_pair_box, v_y_pair_box, v_z_pair_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const ii = v_x_lo + 2 * (i - v_x_lo);
            amrex::Real const rho_u_avg0 = myhalf * (
                rho_u(ii,j,k) * mf_uy_inv(ii,j,0) +
                rho_u(ii,j-1,k) * mf_uy_inv(ii,j-1,0));
            amrex::Real interpx0 = zero;

            if (ii < v_x_hi) {
                amrex::Real const rho_u_avg1 = myhalf * (
                    rho_u(ii+1,j,k) * mf_uy_inv(ii+1,j,0) +
                    rho_u(ii+1,j-1,k) * mf_uy_inv(ii+1,j-1,0));
                amrex::Real interpx1 = zero;
                erf_advection_detail::regular_interpolate_pair<0>(
                    interp_v_horz, interp_v_CEN2, ii, j, k,
                    rho_u_avg0, rho_u_avg1, interpx0, interpx1);
                flx_v_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
                flx_v_arr[0](ii+1,j,k) = rho_u_avg1 * interpx1;
            } else {
                erf_advection_detail::regular_interpolate<0>(
                    interp_v_horz, interp_v_CEN2, ii, j, k, rho_u_avg0, interpx0);
                flx_v_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const jj = v_y_lo + 2 * (j - v_y_lo);
            amrex::Real const rho_vm1 = rho_v(i,jj-1,k);
            amrex::Real const rho_v0 = rho_v(i,jj,k);
            amrex::Real const mf_vm1 = mf_vy_inv(i,jj-1,0);
            amrex::Real const mf_v0 = mf_vy_inv(i,jj,0);
            amrex::Real const rho_v_avg0 = myhalf * (rho_v0 * mf_v0 + rho_vm1 * mf_vm1);
            amrex::Real interpy0 = zero;

            if (jj < v_y_hi) {
                amrex::Real const rho_vp1 = rho_v(i,jj+1,k);
                amrex::Real const mf_vp1 = mf_vy_inv(i,jj+1,0);
                amrex::Real const rho_v_avg1 = myhalf * (rho_vp1 * mf_vp1 + rho_v0 * mf_v0);
                amrex::Real interpy1 = zero;
                erf_advection_detail::regular_interpolate_pair<1>(
                    interp_v_horz, interp_v_CEN2, i, jj, k,
                    rho_v_avg0, rho_v_avg1, interpy0, interpy1);
                flx_v_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
                flx_v_arr[1](i,jj+1,k) = rho_v_avg1 * interpy1;
            } else {
                erf_advection_detail::regular_interpolate<1>(
                    interp_v_horz, interp_v_CEN2, i, jj, k, rho_v_avg0, interpy0);
                flx_v_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const kk = v_z_lo + 2 * (k - v_z_lo);
            amrex::Real const rho_w_avg0 = myhalf * (rho_w(i,j,kk) + rho_w(i,j-1,kk));
            amrex::Real interpz0 = zero;

            if (kk < v_z_hi) {
                amrex::Real const rho_w_avg1 = myhalf * (rho_w(i,j,kk+1) + rho_w(i,j-1,kk+1));
                amrex::Real interpz1 = zero;
                erf_advection_detail::regular_interpolate_pair<2>(
                    interp_v_vert, interp_v_CEN2, i, j, kk,
                    rho_w_avg0, rho_w_avg1, interpz0, interpz1);
                flx_v_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
                flx_v_arr[2](i,j,kk+1) = rho_w_avg1 * interpz1;
            } else {
                erf_advection_detail::regular_interpolate<2>(
                    interp_v_vert, interp_v_CEN2, i, j, kk, rho_w_avg0, interpz0);
                flx_v_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
            }
        });
    } else {
        amrex::ParallelFor(bxy_grown[0], bxy_grown[1], bxy_grown[2],
    // y-momentum: x-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = v_is_regular || (v_is_singlevalued && v_afrac_x(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_u_avg = myhalf * (rho_u(i,j,k) * mf_uy_inv(i,j,0) + rho_u(i,j-1,k) * mf_uy_inv(i,j-1,0));

            int icell = 0;

            if (v_x_stencil_is_regular)
            {
                if (rho_u_avg > zero || rho_u_avg < zero) {
                    icell = ncells_v_horz;
                }
            }
            else if (rho_u_avg>zero)
            {
                for (int ii=0; ii<ncells_v_horz; ++ii) {
                    if (v_cflag(i-ii-1,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }
            else if (rho_u_avg<zero)
            {
                for (int ii=0; ii<ncells_v_horz; ++ii) {
                    if (v_cflag(i+ii,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==ncells_v_horz) {
                interp_v_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                if (icell<=1) {
                    interp_v_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                } else if (icell==2) {
                    interp_v_UPW3.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                }
            }
            flx_v_arr[0](i,j,k) = rho_u_avg * interpx;

        } else {

            flx_v_arr[0](i,j,k) = zero;
        }
    },
    // y-momentum: y-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = v_is_regular || (v_is_singlevalued && v_afrac_y(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) * mf_vy_inv(i,j,0) + rho_v(i,j-1,k) * mf_vy_inv(i,j-1,0));

            int jcell = 0;

            if (v_y_stencil_is_regular)
            {
                if (rho_v_avg > zero || rho_v_avg < zero) {
                    jcell = ncells_v_horz;
                }
            }
            else if (rho_v_avg>zero)
            {
                for (int jj=0; jj<ncells_v_horz; ++jj) {
                    if (v_cflag(i,j-jj-1,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }
            else if (rho_v_avg<zero)
            {
                for (int jj=0; jj<ncells_v_horz; ++jj) {
                    if (v_cflag(i,j+jj,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==ncells_v_horz) {
                interp_v_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                if (jcell<=1) {
                    interp_v_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                } else if (jcell==2) {
                    interp_v_UPW3.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                }
            }
            flx_v_arr[1](i,j,k) = rho_v_avg * interpy;

        } else {

            flx_v_arr[1](i,j,k) = zero;
        }
    },
    // y-momentum: z-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = v_is_regular || (v_is_singlevalued && v_afrac_z(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i,j-1,k));

            int kcell = 0;

            if (v_z_stencil_is_regular)
            {
                if (rho_w_avg > zero || rho_w_avg < zero) {
                    kcell = ncells_v_vert;
                }
            }
            else if (rho_w_avg>zero)
            {
                for (int kk=0; kk<ncells_v_vert; ++kk) {
                    if (v_cflag(i,j,k-kk-1).isCovered()) {
                        break;
                    }
                    kcell++;
                }
            }
            else if (rho_w_avg<zero)
            {
                for (int kk=0; kk<ncells_v_vert; ++kk) {
                    if (v_cflag(i,j,k+kk).isCovered()) {
                        break;
                    }
                    kcell++;
                }
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpz = zero;
            if (kcell==ncells_v_vert) {
                interp_v_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            } else {
                if (kcell<=1) {
                    interp_v_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                } else if (kcell==2) {
                    interp_v_UPW3.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                }
            }
            flx_v_arr[2](i,j,k) = rho_w_avg * interpz;

        } else {

            flx_v_arr[2](i,j,k) = zero;
        }
        });
    }

    // ---------------------------------------------------- //
    // z-momentum
    // ---------------------------------------------------- //

    if (w_has_regular_stencils) {
        amrex::Box const w_x_pair_box = erf_advection_detail::paired_flux_box(bxz_grown[0], 0);
        amrex::Box const w_y_pair_box = erf_advection_detail::paired_flux_box(bxz_grown[1], 1);
        amrex::Box const w_z_pair_box = erf_advection_detail::paired_flux_box(bxz_grown[2], 2);
        int const w_x_lo = bxz_grown[0].smallEnd(0);
        int const w_y_lo = bxz_grown[1].smallEnd(1);
        int const w_z_lo = bxz_grown[2].smallEnd(2);
        int const w_x_hi = bxz_grown[0].bigEnd(0);
        int const w_y_hi = bxz_grown[1].bigEnd(1);
        int const w_z_hi = bxz_grown[2].bigEnd(2);

        amrex::ParallelFor(w_x_pair_box, w_y_pair_box, w_z_pair_box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const ii = w_x_lo + 2 * (i - w_x_lo);
            amrex::Real const rho_u_avg0 = myhalf * (
                rho_u(ii,j,k) + rho_u(ii,j,k-1)) * mf_ux_inv(ii,j,0);
            amrex::Real interpx0 = zero;

            if (ii < w_x_hi) {
                amrex::Real const rho_u_avg1 = myhalf * (
                    rho_u(ii+1,j,k) + rho_u(ii+1,j,k-1)) * mf_ux_inv(ii+1,j,0);
                amrex::Real interpx1 = zero;
                erf_advection_detail::regular_interpolate_pair<0>(
                    interp_w_horz, interp_w_CEN2, ii, j, k,
                    rho_u_avg0, rho_u_avg1, interpx0, interpx1);
                flx_w_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
                flx_w_arr[0](ii+1,j,k) = rho_u_avg1 * interpx1;
            } else {
                erf_advection_detail::regular_interpolate<0>(
                    interp_w_horz, interp_w_CEN2, ii, j, k, rho_u_avg0, interpx0);
                flx_w_arr[0](ii,j,k) = rho_u_avg0 * interpx0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const jj = w_y_lo + 2 * (j - w_y_lo);
            amrex::Real const rho_v_avg0 = myhalf * (
                rho_v(i,jj,k) + rho_v(i,jj,k-1)) * mf_vy_inv(i,jj,0);
            amrex::Real interpy0 = zero;

            if (jj < w_y_hi) {
                amrex::Real const rho_v_avg1 = myhalf * (
                    rho_v(i,jj+1,k) + rho_v(i,jj+1,k-1)) * mf_vy_inv(i,jj+1,0);
                amrex::Real interpy1 = zero;
                erf_advection_detail::regular_interpolate_pair<1>(
                    interp_w_horz, interp_w_CEN2, i, jj, k,
                    rho_v_avg0, rho_v_avg1, interpy0, interpy1);
                flx_w_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
                flx_w_arr[1](i,jj+1,k) = rho_v_avg1 * interpy1;
            } else {
                erf_advection_detail::regular_interpolate<1>(
                    interp_w_horz, interp_w_CEN2, i, jj, k, rho_v_avg0, interpy0);
                flx_w_arr[1](i,jj,k) = rho_v_avg0 * interpy0;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            int const kk = w_z_lo + 2 * (k - w_z_lo);
            bool const has_second = kk < w_z_hi;
            bool const direct0 = kk == lo_z_face || kk == hi_z_face+1;
            bool const direct1 = has_second &&
                                 (kk+1 == lo_z_face || kk+1 == hi_z_face+1);

            if (direct0) {
                flx_w_arr[2](i,j,kk) = rho_w(i,j,kk) * w(i,j,kk);
            }
            if (direct1) {
                flx_w_arr[2](i,j,kk+1) = rho_w(i,j,kk+1) * w(i,j,kk+1);
            }

            if (!direct0 && has_second && !direct1) {
                amrex::Real const rho_wm1 = rho_w(i,j,kk-1);
                amrex::Real const rho_w0 = rho_w(i,j,kk);
                amrex::Real const rho_wp1 = rho_w(i,j,kk+1);
                amrex::Real const rho_w_avg0 = myhalf * (rho_w0 + rho_wm1);
                amrex::Real const rho_w_avg1 = myhalf * (rho_wp1 + rho_w0);
                int const kcell0 = erf_advection_detail::regular_w_vertical_cells<
                    vertical_stencil_size>(kk, lo_z_face, hi_z_face, rho_w_avg0);
                int const kcell1 = erf_advection_detail::regular_w_vertical_cells<
                    vertical_stencil_size>(kk+1, lo_z_face, hi_z_face, rho_w_avg1);
                amrex::Real interpz0 = zero;
                amrex::Real interpz1 = zero;
                erf_advection_detail::regular_w_vertical_interpolate_pair<
                    vertical_stencil_size>(interp_w_vert, interp_w_CEN2, interp_w_UPW3,
                                           i, j, kk, kcell0, kcell1,
                                           rho_w_avg0, rho_w_avg1, interpz0, interpz1);
                flx_w_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
                flx_w_arr[2](i,j,kk+1) = rho_w_avg1 * interpz1;
            } else {
                if (!direct0) {
                    amrex::Real const rho_w_avg0 = myhalf * (
                        rho_w(i,j,kk) + rho_w(i,j,kk-1));
                    int const kcell0 = erf_advection_detail::regular_w_vertical_cells<
                        vertical_stencil_size>(kk, lo_z_face, hi_z_face, rho_w_avg0);
                    amrex::Real interpz0 = zero;
                    erf_advection_detail::regular_w_vertical_interpolate<
                        vertical_stencil_size>(interp_w_vert, interp_w_CEN2, interp_w_UPW3,
                                               i, j, kk, kcell0, rho_w_avg0, interpz0);
                    flx_w_arr[2](i,j,kk) = rho_w_avg0 * interpz0;
                }
                if (has_second && !direct1) {
                    amrex::Real const rho_w_avg1 = myhalf * (
                        rho_w(i,j,kk+1) + rho_w(i,j,kk));
                    int const kcell1 = erf_advection_detail::regular_w_vertical_cells<
                        vertical_stencil_size>(kk+1, lo_z_face, hi_z_face, rho_w_avg1);
                    amrex::Real interpz1 = zero;
                    erf_advection_detail::regular_w_vertical_interpolate<
                        vertical_stencil_size>(interp_w_vert, interp_w_CEN2, interp_w_UPW3,
                                               i, j, kk+1, kcell1, rho_w_avg1, interpz1);
                    flx_w_arr[2](i,j,kk+1) = rho_w_avg1 * interpz1;
                }
            }
        });
    } else {
        amrex::ParallelFor(bxz_grown[0], bxz_grown[1], bxz_grown[2],
    // z-momentum: x-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = w_is_regular || (w_is_singlevalued && w_afrac_x(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_u_avg = myhalf * (rho_u(i,j,k) + rho_u(i,j,k-1)) * mf_ux_inv(i,j,0);

            int icell = 0;

            if (w_x_stencil_is_regular)
            {
                if (rho_u_avg > zero || rho_u_avg < zero) {
                    icell = ncells_w_horz;
                }
            }
            else if (rho_u_avg>zero)
            {
                for (int ii=0; ii<ncells_w_horz; ++ii) {
                    if (w_cflag(i-ii-1,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }
            else if (rho_u_avg<zero)
            {
                for (int ii=0; ii<ncells_w_horz; ++ii) {
                    if (w_cflag(i+ii,j,k).isCovered()) {
                        break;
                    }
                    icell++;
                }
            }

            // Interpolate w using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==ncells_w_horz) {
                interp_w_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                if (icell<=1) {
                    interp_w_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                } else if (icell==2) {
                    interp_w_UPW3.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
                }
            }
            flx_w_arr[0](i,j,k) = rho_u_avg * interpx;

        } else {

            flx_w_arr[0](i,j,k) = zero;
        }
    },
    // z-momentum: y-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = w_is_regular || (w_is_singlevalued && w_afrac_y(i,j,k) > zero);

        if ( not_covered ) {

            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) + rho_v(i,j,k-1)) * mf_vy_inv(i,j,0);

            int jcell = 0;

            if (w_y_stencil_is_regular)
            {
                if (rho_v_avg > zero || rho_v_avg < zero) {
                    jcell = ncells_w_horz;
                }
            }
            else if (rho_v_avg>zero)
            {
                for (int jj=0; jj<ncells_w_horz; ++jj) {
                    if (w_cflag(i,j-jj-1,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }
            else if (rho_v_avg<zero)
            {
                for (int jj=0; jj<ncells_w_horz; ++jj) {
                    if (w_cflag(i,j+jj,k).isCovered()) {
                        break;
                    }
                    jcell++;
                }
            }

            // Interpolate w using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==ncells_w_horz) {
                interp_w_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                if (jcell<=1) {
                    interp_w_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                } else if (jcell==2) {
                    interp_w_UPW3.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
                }
            }
            flx_w_arr[1](i,j,k) = rho_v_avg * interpy;

        } else {

            flx_w_arr[1](i,j,k) = zero;
        }
    },
    // z-momentum: z-face
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        bool not_covered = w_is_regular || (w_is_singlevalued && w_afrac_z(i,j,k) > zero);

        if ( not_covered ) {

            if (k==lo_z_face || k==hi_z_face+1) {

                flx_w_arr[2](i,j,k) = rho_w(i,j,k) * w(i,j,k);

            } else {
                amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i,j,k-1));

                int kcell = 0;

                if (w_z_stencil_is_regular)
                {
                    if (rho_w_avg > zero || rho_w_avg < zero) {
                        kcell = ncells_w_vert;
                    }
                }
                else if (rho_w_avg>zero)
                {
                    for (int kk=0; kk<ncells_w_vert; ++kk) {
                        if (w_cflag(i,j,k-kk-1).isCovered()) {
                            break;
                        }
                        kcell++;
                    }
                }
                else if (rho_w_avg<zero)
                {
                    for (int kk=0; kk<ncells_w_vert; ++kk) {
                        if (w_cflag(i,j,k+kk).isCovered()) {
                            break;
                        }
                        kcell++;
                    }
                }

                // Adjust available cell numbers near the top and bottom boundaries

                if (k == hi_z_face || k == lo_z_face+1) {
                    kcell=std::min(kcell,1);
                } else if (k == hi_z_face-1 || k == lo_z_face+2 ) {
                    kcell=std::min(kcell,2);
                }

                // Interpolate w using the highest-order scheme

                amrex::Real interpz = zero;
                if (kcell==ncells_w_vert) {
                    interp_w_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                } else {
                    if (kcell<=1) {
                        interp_w_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                    } else if (kcell==2) {
                        interp_w_UPW3.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                    }
                }
                flx_w_arr[2](i,j,k) = rho_w_avg * interpz;
            } // lo_z_face, hi_z_face

        } else {

            flx_w_arr[2](i,j,k) = zero;
        }
        });
    }

}

/**
 * \brief Dispatch EB momentum flux construction on the vertical advection scheme.
 * \tparam InterpType_H Horizontal interpolation functor selected by the caller.
 */
template<typename InterpType_H>
void
EBAdvectionSrcForMomVert (const amrex::Vector<amrex::Box>& bxx_grown,
                        const amrex::Vector<amrex::Box>& bxy_grown,
                        const amrex::Vector<amrex::Box>& bxz_grown,
                        const amrex::Array4<const amrex::Real>& rho_u,
                        const amrex::Array4<const amrex::Real>& rho_v,
                        const amrex::Array4<const amrex::Real>& rho_w,
                        const amrex::Array4<const amrex::Real>& u,
                        const amrex::Array4<const amrex::Real>& v,
                        const amrex::Array4<const amrex::Real>& w,
                        const amrex::FabType u_type,
                        const amrex::FabType v_type,
                        const amrex::FabType w_type,
                        const amrex::EBCellFlagFab& u_flag_fab,
                        const amrex::EBCellFlagFab& v_flag_fab,
                        const amrex::EBCellFlagFab& w_flag_fab,
                        const amrex::Array4<const amrex::EBCellFlag>& u_cflag,
                        const amrex::Array4<const amrex::Real>& u_afrac_x,
                        const amrex::Array4<const amrex::Real>& u_afrac_y,
                        const amrex::Array4<const amrex::Real>& u_afrac_z,
                        const amrex::Array4<const amrex::EBCellFlag>& v_cflag,
                        const amrex::Array4<const amrex::Real>& v_afrac_x,
                        const amrex::Array4<const amrex::Real>& v_afrac_y,
                        const amrex::Array4<const amrex::Real>& v_afrac_z,
                        const amrex::Array4<const amrex::EBCellFlag>& w_cflag,
                        const amrex::Array4<const amrex::Real>& w_afrac_x,
                        const amrex::Array4<const amrex::Real>& w_afrac_y,
                        const amrex::Array4<const amrex::Real>& w_afrac_z,
                        const amrex::Array4<const amrex::Real>& mf_ux_inv,
                        const amrex::Array4<const amrex::Real>& mf_vx_inv,
                        const amrex::Array4<const amrex::Real>& mf_uy_inv,
                        const amrex::Array4<const amrex::Real>& mf_vy_inv,
                        const amrex::Real upw_frac_h,
                        const amrex::Real upw_frac_v,
                        const AdvType vert_adv_type,
                              amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_u_arr,
                              amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_v_arr,
                              amrex::GpuArray<amrex::Array4<amrex::Real>, AMREX_SPACEDIM>& flx_w_arr,
                        const int lo_z_face, const int hi_z_face)
{
    if (vert_adv_type == AdvType::Centered_2nd) {
        EBAdvectionSrcForMomWrapper<InterpType_H,CENTERED2,UPWINDALL>(
                                bxx_grown, bxy_grown, bxz_grown,
                                rho_u, rho_v, rho_w, u, v, w,
                                u_type, v_type, w_type,
                                u_flag_fab, v_flag_fab, w_flag_fab,
                                u_cflag, u_afrac_x, u_afrac_y, u_afrac_z,
                                v_cflag, v_afrac_x, v_afrac_y, v_afrac_z,
                                w_cflag, w_afrac_x, w_afrac_y, w_afrac_z,
                                mf_ux_inv, mf_vx_inv,
                                mf_uy_inv, mf_vy_inv,
                                upw_frac_h, upw_frac_v, vert_adv_type,
                                flx_u_arr, flx_v_arr, flx_w_arr,
                                lo_z_face, hi_z_face);
    } else if (vert_adv_type == AdvType::Upwind_3rd) {
        EBAdvectionSrcForMomWrapper<InterpType_H,UPWIND3,UPWINDALL>(
                                bxx_grown, bxy_grown, bxz_grown,
                                rho_u, rho_v, rho_w, u, v, w,
                                u_type, v_type, w_type,
                                u_flag_fab, v_flag_fab, w_flag_fab,
                                u_cflag, u_afrac_x, u_afrac_y, u_afrac_z,
                                v_cflag, v_afrac_x, v_afrac_y, v_afrac_z,
                                w_cflag, w_afrac_x, w_afrac_y, w_afrac_z,
                                mf_ux_inv, mf_vx_inv,
                                mf_uy_inv, mf_vy_inv,
                                upw_frac_h, upw_frac_v, vert_adv_type,
                                flx_u_arr, flx_v_arr, flx_w_arr,
                                lo_z_face, hi_z_face);
    } else if (vert_adv_type == AdvType::Centered_4th) {
        EBAdvectionSrcForMomWrapper<InterpType_H,CENTERED4,UPWINDALL>(
                                bxx_grown, bxy_grown, bxz_grown,
                                rho_u, rho_v, rho_w, u, v, w,
                                u_type, v_type, w_type,
                                u_flag_fab, v_flag_fab, w_flag_fab,
                                u_cflag, u_afrac_x, u_afrac_y, u_afrac_z,
                                v_cflag, v_afrac_x, v_afrac_y, v_afrac_z,
                                w_cflag, w_afrac_x, w_afrac_y, w_afrac_z,
                                mf_ux_inv, mf_vx_inv,
                                mf_uy_inv, mf_vy_inv,
                                upw_frac_h, upw_frac_v, vert_adv_type,
                                flx_u_arr, flx_v_arr, flx_w_arr,
                                lo_z_face, hi_z_face);
    } else if (vert_adv_type == AdvType::Upwind_5th) {
        EBAdvectionSrcForMomWrapper<InterpType_H,UPWIND5,UPWINDALL>(
                                bxx_grown, bxy_grown, bxz_grown,
                                rho_u, rho_v, rho_w, u, v, w,
                                u_type, v_type, w_type,
                                u_flag_fab, v_flag_fab, w_flag_fab,
                                u_cflag, u_afrac_x, u_afrac_y, u_afrac_z,
                                v_cflag, v_afrac_x, v_afrac_y, v_afrac_z,
                                w_cflag, w_afrac_x, w_afrac_y, w_afrac_z,
                                mf_ux_inv, mf_vx_inv,
                                mf_uy_inv, mf_vy_inv,
                                upw_frac_h, upw_frac_v, vert_adv_type,
                                flx_u_arr, flx_v_arr, flx_w_arr,
                                lo_z_face, hi_z_face);
    } else if (vert_adv_type == AdvType::Centered_6th) {
        EBAdvectionSrcForMomWrapper<InterpType_H,CENTERED6,UPWINDALL>(
                                bxx_grown, bxy_grown, bxz_grown,
                                rho_u, rho_v, rho_w, u, v, w,
                                u_type, v_type, w_type,
                                u_flag_fab, v_flag_fab, w_flag_fab,
                                u_cflag, u_afrac_x, u_afrac_y, u_afrac_z,
                                v_cflag, v_afrac_x, v_afrac_y, v_afrac_z,
                                w_cflag, w_afrac_x, w_afrac_y, w_afrac_z,
                                mf_ux_inv, mf_vx_inv,
                                mf_uy_inv, mf_vy_inv,
                                upw_frac_h, upw_frac_v, vert_adv_type,
                                flx_u_arr, flx_v_arr, flx_w_arr,
                                lo_z_face, hi_z_face);
    } else {
        AMREX_ASSERT_WITH_MESSAGE(false, "Unknown advection scheme!");
    }
}

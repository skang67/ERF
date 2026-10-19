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

/**
 * \brief Count consecutive noncovered upwind cells for one flux direction.
 *
 * Positive flow starts one cell below the flux index and steps backward;
 * negative flow starts at the flux index and steps forward.  Zero and
 * unordered flow retain zero availability.
 */
template<int Dir, int StencilSize>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
int
upwind_cell_availability (int i, int j, int k,
                          amrex::Real mass_flux,
                          const amrex::Array4<const amrex::EBCellFlag>& flag) noexcept
{
    static_assert(Dir >= 0 && Dir < AMREX_SPACEDIM);
    static_assert(StencilSize > 0);

    bool const positive = mass_flux > zero;
    if (!positive && !(mass_flux < zero)) {
        return 0;
    }

    int index;
    if constexpr (Dir == 0) {
        index = i;
    } else if constexpr (Dir == 1) {
        index = j;
    } else {
        index = k;
    }
    index -= static_cast<int>(positive);

    int const step = positive ? -1 : 1;
    int available = 0;
    for (int n = 0; n < StencilSize; ++n) {
        bool covered;
        if constexpr (Dir == 0) {
            covered = flag(index,j,k).isCovered();
        } else if constexpr (Dir == 1) {
            covered = flag(i,index,k).isCovered();
        } else {
            covered = flag(i,j,index).isCovered();
        }
        if (covered) {
            break;
        }
        ++available;
        index += step;
    }
    return available;
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
        amrex::ParallelFor(bxx_grown[0], bxx_grown[1], bxx_grown[2],
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_u_avg = myhalf * (rho_u(i-1,j,k) * mf_ux_inv(i-1,j,0) + rho_u(i,j,k) * mf_ux_inv(i,j,0));
            amrex::Real interpx = zero;
            if (rho_u_avg > zero || rho_u_avg < zero) {
                interp_u_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                interp_u_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            }
            flx_u_arr[0](i,j,k) = rho_u_avg * interpx;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) * mf_vx_inv(i,j,0) + rho_v(i-1,j,k) * mf_vx_inv(i-1,j,0));
            amrex::Real interpy = zero;
            if (rho_v_avg > zero || rho_v_avg < zero) {
                interp_u_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                interp_u_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            }
            flx_u_arr[1](i,j,k) = rho_v_avg * interpy;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i-1,j,k));
            amrex::Real interpz = zero;
            if (rho_w_avg > zero || rho_w_avg < zero) {
                interp_u_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            } else {
                interp_u_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            }
            flx_u_arr[2](i,j,k) = rho_w_avg * interpz;
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
                    icell = horizontal_stencil_size;
                }
            }
            else
            {
                icell = erf_advection_detail::upwind_cell_availability<0,horizontal_stencil_size>(
                    i, j, k, rho_u_avg, u_cflag);
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==horizontal_stencil_size) {
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
                    jcell = horizontal_stencil_size;
                }
            }
            else
            {
                jcell = erf_advection_detail::upwind_cell_availability<1,horizontal_stencil_size>(
                    i, j, k, rho_v_avg, u_cflag);
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==horizontal_stencil_size) {
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
                    kcell = vertical_stencil_size;
                }
            }
            else
            {
                kcell = erf_advection_detail::upwind_cell_availability<2,vertical_stencil_size>(
                    i, j, k, rho_w_avg, u_cflag);
            }

            // Interpolate u using the highest-order scheme

            amrex::Real interpz = zero;
            if (kcell==vertical_stencil_size) {
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
        amrex::ParallelFor(bxy_grown[0], bxy_grown[1], bxy_grown[2],
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_u_avg = myhalf * (rho_u(i,j,k) * mf_uy_inv(i,j,0) + rho_u(i,j-1,k) * mf_uy_inv(i,j-1,0));
            amrex::Real interpx = zero;
            if (rho_u_avg > zero || rho_u_avg < zero) {
                interp_v_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                interp_v_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            }
            flx_v_arr[0](i,j,k) = rho_u_avg * interpx;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) * mf_vy_inv(i,j,0) + rho_v(i,j-1,k) * mf_vy_inv(i,j-1,0));
            amrex::Real interpy = zero;
            if (rho_v_avg > zero || rho_v_avg < zero) {
                interp_v_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                interp_v_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            }
            flx_v_arr[1](i,j,k) = rho_v_avg * interpy;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i,j-1,k));
            amrex::Real interpz = zero;
            if (rho_w_avg > zero || rho_w_avg < zero) {
                interp_v_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            } else {
                interp_v_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
            }
            flx_v_arr[2](i,j,k) = rho_w_avg * interpz;
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
                    icell = horizontal_stencil_size;
                }
            }
            else
            {
                icell = erf_advection_detail::upwind_cell_availability<0,horizontal_stencil_size>(
                    i, j, k, rho_u_avg, v_cflag);
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==horizontal_stencil_size) {
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
                    jcell = horizontal_stencil_size;
                }
            }
            else
            {
                jcell = erf_advection_detail::upwind_cell_availability<1,horizontal_stencil_size>(
                    i, j, k, rho_v_avg, v_cflag);
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==horizontal_stencil_size) {
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
                    kcell = vertical_stencil_size;
                }
            }
            else
            {
                kcell = erf_advection_detail::upwind_cell_availability<2,vertical_stencil_size>(
                    i, j, k, rho_w_avg, v_cflag);
            }

            // Interpolate v using the highest-order scheme

            amrex::Real interpz = zero;
            if (kcell==vertical_stencil_size) {
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
        amrex::ParallelFor(bxz_grown[0], bxz_grown[1], bxz_grown[2],
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_u_avg = myhalf * (rho_u(i,j,k) + rho_u(i,j,k-1)) * mf_ux_inv(i,j,0);
            amrex::Real interpx = zero;
            if (rho_u_avg > zero || rho_u_avg < zero) {
                interp_w_horz.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            } else {
                interp_w_CEN2.InterpolateInX(i,j,k,0,interpx,rho_u_avg);
            }
            flx_w_arr[0](i,j,k) = rho_u_avg * interpx;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            amrex::Real rho_v_avg = myhalf * (rho_v(i,j,k) + rho_v(i,j,k-1)) * mf_vy_inv(i,j,0);
            amrex::Real interpy = zero;
            if (rho_v_avg > zero || rho_v_avg < zero) {
                interp_w_horz.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            } else {
                interp_w_CEN2.InterpolateInY(i,j,k,0,interpy,rho_v_avg);
            }
            flx_w_arr[1](i,j,k) = rho_v_avg * interpy;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (k==lo_z_face || k==hi_z_face+1) {
                flx_w_arr[2](i,j,k) = rho_w(i,j,k) * w(i,j,k);
            } else {
                amrex::Real rho_w_avg = myhalf * (rho_w(i,j,k) + rho_w(i,j,k-1));
                int kcell = 0;
                if (rho_w_avg > zero || rho_w_avg < zero) {
                    kcell = vertical_stencil_size;
                }

                if (k == hi_z_face || k == lo_z_face+1) {
                    kcell=std::min(kcell,1);
                } else if (k == hi_z_face-1 || k == lo_z_face+2 ) {
                    kcell=std::min(kcell,2);
                }

                amrex::Real interpz = zero;
                if (kcell==vertical_stencil_size) {
                    interp_w_vert.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                } else {
                    if (kcell<=1) {
                        interp_w_CEN2.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                    } else if (kcell==2) {
                        interp_w_UPW3.InterpolateInZ(i,j,k,0,interpz,rho_w_avg);
                    }
                }
                flx_w_arr[2](i,j,k) = rho_w_avg * interpz;
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
                    icell = horizontal_stencil_size;
                }
            }
            else
            {
                icell = erf_advection_detail::upwind_cell_availability<0,horizontal_stencil_size>(
                    i, j, k, rho_u_avg, w_cflag);
            }

            // Interpolate w using the highest-order scheme

            amrex::Real interpx = zero;
            if (icell==horizontal_stencil_size) {
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
                    jcell = horizontal_stencil_size;
                }
            }
            else
            {
                jcell = erf_advection_detail::upwind_cell_availability<1,horizontal_stencil_size>(
                    i, j, k, rho_v_avg, w_cflag);
            }

            // Interpolate w using the highest-order scheme

            amrex::Real interpy = zero;
            if (jcell==horizontal_stencil_size) {
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
                        kcell = vertical_stencil_size;
                    }
                }
                else
                {
                    kcell = erf_advection_detail::upwind_cell_availability<2,vertical_stencil_size>(
                        i, j, k, rho_w_avg, w_cflag);
                }

                // Adjust available cell numbers near the top and bottom boundaries

                if (k == hi_z_face || k == lo_z_face+1) {
                    kcell=std::min(kcell,1);
                } else if (k == hi_z_face-1 || k == lo_z_face+2 ) {
                    kcell=std::min(kcell,2);
                }

                // Interpolate w using the highest-order scheme

                amrex::Real interpz = zero;
                if (kcell==vertical_stencil_size) {
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

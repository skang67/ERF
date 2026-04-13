
#include <ERF_TI_fast_headers.H>
#include <ERF_EBRedistribute.H>

using namespace amrex;

/**
 * Function for computing the fast RHS with Embedded Boundaries (EB) and flat/constant-dz mesh.
 *
 * This is the EB analogue of erf_substep_NS.  All cell/face updates are guarded by
 * the appropriate EB volume-fraction arrays so that covered cells and faces are not
 * modified.  The caller (acoustic_substepping_fun in ERF_TI_substep_fun.H) is
 * responsible for calling redistribute_term on the per-substep increment afterwards.
 *
 * @param[in   ]  step  which fast time step within each Runge-Kutta step
 * @param[in   ]  nrk   which Runge-Kutta step
 * @param[in   ]  level level of resolution
 * @param[in   ]  finest_level finest level of resolution
 * @param[in   ]  S_slow_rhs slow RHS computed in erf_slow_rhs_pre
 * @param[in   ]  S_prev   if step == 0 this is S_old, else the previous fast solution
 * @param[in   ]  S_stage_data solution at previous RK stage
 * @param[in   ]  S_stage_prim primitive variables at previous RK stage
 * @param[in   ]  pi_stage   Exner function at previous RK stage
 * @param[in   ]  fast_coeffs coefficients for the tridiagonal solve
 * @param[  out]  S_data    current solution
 * @param[inout]  lagged_delta_rt
 * @param[inout]  avg_xmom: time-averaged x-momentum for updating slow variables
 * @param[inout]  avg_ymom: time-averaged y-momentum for updating slow variables
 * @param[inout]  avg_zmom: time-averaged z-momentum for updating slow variables
 * @param[in   ]  cc_src  source terms for conserved variables
 * @param[in   ]  xmom_src source terms for x-momentum
 * @param[in   ]  ymom_src source terms for y-momentum
 * @param[in   ]  zmom_src source terms for z-momentum
 * @param[in   ]  geom container for geometric information
 * @param[in   ]  gravity magnitude of gravity
 * @param[in   ]  stretched_dz_d cell thicknesses (constant-dz path uses dz_ptr[k] = 1/dzi)
 * @param[in   ]  dtau fast time step
 * @param[in   ]  beta_s coefficient for implicit fraction of the solve
 * @param[in   ]  facinv inverse factor for time-averaging the momenta
 * @param[in   ]  mapfac vector of map factors
 * @param[inout]  fr_as_crse YAFluxRegister at level l / l+1 interface
 * @param[inout]  fr_as_fine YAFluxRegister at level l-1 / l interface
 * @param[in   ]  l_use_moisture whether moisture is active
 * @param[in   ]  l_reflux should we add fluxes to the FluxRegisters?
 * @param[in   ]  sinesq_stag_d Rayleigh damping profile (nullptr if not used)
 * @param[in   ]  l_damp_coef Rayleigh damping coefficient
 * @param[in   ]  ebfact EB factories (cell-centered and staggered)
 */
void erf_substep_EB (int step, int nrk,
                     int level, int finest_level,
                     Vector<MultiFab>& S_slow_rhs,
                     const Vector<MultiFab>& S_prev,
                     Vector<MultiFab>& S_stage_data,
                     const  MultiFab & S_stage_prim,
                     const  MultiFab & qt,
                     const  MultiFab & pi_stage,
                     const  MultiFab & fast_coeffs,
                     Vector<MultiFab>& S_data,
                     MultiFab& lagged_delta_rt,
                     MultiFab& avg_xmom,
                     MultiFab& avg_ymom,
                     MultiFab& avg_zmom,
                     const MultiFab& cc_src,
                     const MultiFab& xmom_src,
                     const MultiFab& ymom_src,
                     const MultiFab& zmom_src,
                     const Geometry geom,
                     const Real gravity,
                     amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,
                     const Real dtau, const Real beta_s,
                     const Real facinv,
                     Vector<std::unique_ptr<MultiFab>>& mapfac,
                     YAFluxRegister* fr_as_crse,
                     YAFluxRegister* fr_as_fine,
                     bool l_use_moisture,
                     bool l_reflux,
                     const amrex::Real* sinesq_stag_d,
                     const Real l_damp_coef,
                     const eb_& ebfact)
{
    //
    // NOTE: for step > 0, S_data and S_prev point to the same MultiFab data!!
    //

    BL_PROFILE_REGION("erf_substep_EB()");

    Real beta_1 = myhalf * (one - beta_s);  // multiplies explicit terms
    Real beta_2 = myhalf * (one + beta_s);  // multiplies implicit terms

    Real beta_d = Real(0.1);

    Real RvOverRd = R_v / R_d;

    bool l_rayleigh_impl_for_w = (sinesq_stag_d != nullptr);

    const Real* dx = geom.CellSize();
    const GpuArray<Real, AMREX_SPACEDIM> dxInv = geom.InvCellSizeArray();

    Real dxi = dxInv[0];
    Real dyi = dxInv[1];

    auto dz_ptr = stretched_dz_d.data();

    const auto& ba = S_stage_data[IntVars::cons].boxArray();
    const auto& dm = S_stage_data[IntVars::cons].DistributionMap();

    MultiFab Delta_rho_theta(        ba                , dm, 1, 1);
    MultiFab Delta_rho_w    (convert(ba,IntVect(0,0,1)), dm, 1, IntVect(1,1,0));

    MultiFab     coeff_A_mf(fast_coeffs, make_alias, 0, 1);
    MultiFab inv_coeff_B_mf(fast_coeffs, make_alias, 1, 1);
    MultiFab     coeff_C_mf(fast_coeffs, make_alias, 2, 1);
    MultiFab     coeff_P_mf(fast_coeffs, make_alias, 3, 1);
    MultiFab     coeff_Q_mf(fast_coeffs, make_alias, 4, 1);

    const    Array<Real,AMREX_SPACEDIM> grav{zero, zero, -gravity};
    const GpuArray<Real,AMREX_SPACEDIM> grav_gpu{grav[0], grav[1], grav[2]};

    MultiFab extrap(S_data[IntVars::cons].boxArray(),S_data[IntVars::cons].DistributionMap(),1,1);
    MultiFab temp_rhs(S_stage_data[IntVars::zmom].boxArray(),S_stage_data[IntVars::zmom].DistributionMap(),2,0);
    MultiFab temp_cur_xmom(S_stage_data[IntVars::xmom].boxArray(),S_stage_data[IntVars::xmom].DistributionMap(),1,0);
    MultiFab temp_cur_ymom(S_stage_data[IntVars::ymom].boxArray(),S_stage_data[IntVars::ymom].DistributionMap(),1,0);

    // Initialize temp_cur momenta to stage values so covered faces are unmodified
    MultiFab::Copy(temp_cur_xmom, S_stage_data[IntVars::xmom], 0, 0, 1, 0);
    MultiFab::Copy(temp_cur_ymom, S_stage_data[IntVars::ymom], 0, 0, 1, 0);

    AMREX_ALWAYS_ASSERT(nrk > 0 || step == 0);

    // *************************************************************************
    // First set up Delta_rho_theta and Delta_rho_w
    // *************************************************************************
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Array4<const Real>& prev_cons  = S_prev[IntVars::cons].const_array(mfi);
        const Array4<const Real>& prev_zmom  = S_prev[IntVars::zmom].const_array(mfi);

        const Array4<const Real>& stage_cons = S_stage_data[IntVars::cons].const_array(mfi);
        const Array4<const Real>& stage_zmom = S_stage_data[IntVars::zmom].const_array(mfi);

        const Array4<Real>& prev_drho_w     = Delta_rho_w.array(mfi);
        const Array4<Real>& prev_drho_theta = Delta_rho_theta.array(mfi);
        const Array4<Real>& lagged_arr      = lagged_delta_rt.array(mfi);
        const Array4<Real>& theta_extrap    = extrap.array(mfi);
        const Array4<const Real>& prim      = S_stage_prim.const_array(mfi);

        // EB: cell-centered volume fraction (used to mask covered cells)
        const Array4<const EBCellFlag>& flag_c = (ebfact.get_const_factory())->getMultiEBCellFlagFab()[mfi].const_array();

        Box gbx = mfi.growntilebox(1);
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // EB: skip covered cells
            if (flag_c(i,j,k).isCovered()) {

                theta_extrap(i,j,k) = zero;
                lagged_arr(i,j,k)   = zero;
                prev_drho_theta(i,j,k) = zero;

            } else {

                prev_drho_theta(i,j,k) = prev_cons(i,j,k,RhoTheta_comp) - stage_cons(i,j,k,RhoTheta_comp);

                if (step == 0) {
                    theta_extrap(i,j,k) = prev_drho_theta(i,j,k);
                } else {
                    theta_extrap(i,j,k) = prev_drho_theta(i,j,k) + beta_d *
                    ( prev_drho_theta(i,j,k) - lagged_arr(i,j,k) );
                }

                Real qv = (l_use_moisture) ? prim(i,j,k,PrimQ1_comp) : zero;
                theta_extrap(i,j,k) *= (one + RvOverRd*qv);

                lagged_arr(i,j,k) = prev_drho_theta(i,j,k);
            }
        });

        Box gtbz = mfi.nodaltilebox(2);
        gtbz.grow(IntVect(1,1,0));

        // EB: z-face cell flag
        const Array4<const EBCellFlag>& flag_w = (ebfact.get_w_const_factory())->getMultiEBCellFlagFab()[mfi].const_array();

        ParallelFor(gtbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            // EB: set zero on covered z-faces
            if (flag_w(i,j,k).isCovered()) {
                prev_drho_w(i,j,k) = zero;
            } else {
                prev_drho_w(i,j,k) = prev_zmom(i,j,k) - stage_zmom(i,j,k);
            }
        });
    } // mfi

    // *************************************************************************
    // Update x- and y-momenta (EB-guarded)
    // *************************************************************************
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box tbx = mfi.nodaltilebox(0);
        Box tby = mfi.nodaltilebox(1);

        const Array4<Real const>& xmom_src_arr = xmom_src.const_array(mfi);
        const Array4<Real const>& ymom_src_arr = ymom_src.const_array(mfi);

        const Array4<const Real>& stage_xmom = S_stage_data[IntVars::xmom].const_array(mfi);
        const Array4<const Real>& stage_ymom = S_stage_data[IntVars::ymom].const_array(mfi);
        const Array4<const Real>& qt_arr     = qt.const_array(mfi);

        const Array4<const Real>& slow_rhs_rho_u = S_slow_rhs[IntVars::xmom].const_array(mfi);
        const Array4<const Real>& slow_rhs_rho_v = S_slow_rhs[IntVars::ymom].const_array(mfi);

        const Array4<Real>& temp_cur_xmom_arr = temp_cur_xmom.array(mfi);
        const Array4<Real>& temp_cur_ymom_arr = temp_cur_ymom.array(mfi);

        const Array4<const Real>& prev_xmom = S_prev[IntVars::xmom].const_array(mfi);
        const Array4<const Real>& prev_ymom = S_prev[IntVars::ymom].const_array(mfi);

        const Array4<Real>& avg_xmom_arr = avg_xmom.array(mfi);
        const Array4<Real>& avg_ymom_arr = avg_ymom.array(mfi);

        const Array4<const Real>& pi_stage_ca = pi_stage.const_array(mfi);
        const Array4<Real>& theta_extrap = extrap.array(mfi);

        // Map factors
        const Array4<const Real>& mf_ux = mapfac[MapFacType::u_x]->const_array(mfi);
        const Array4<const Real>& mf_vy = mapfac[MapFacType::v_y]->const_array(mfi);

        // EB: staggered volume fractions for x- and y-momentum faces
        const Array4<const EBCellFlag>& flag_u = (ebfact.get_u_const_factory())->getMultiEBCellFlagFab()[mfi].const_array();
        const Array4<const EBCellFlag>& flag_v = (ebfact.get_v_const_factory())->getMultiEBCellFlagFab()[mfi].const_array();

        // *********************************************************************
        // Define updates in the RHS of {x, y, z}-momentum equations
        // *********************************************************************
        if (nrk == 0 and step == 0) {
            ParallelFor(tbx, tby,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // EB: skip covered x-faces
                if (flag_u(i,j,k).isCovered()) {
                    temp_cur_xmom_arr(i,j,k) = stage_xmom(i,j,k);
                } else {
                    Real new_drho_u = dtau * slow_rhs_rho_u(i,j,k) + dtau * xmom_src_arr(i,j,k);
                    avg_xmom_arr(i,j,k) += facinv * new_drho_u;
                    temp_cur_xmom_arr(i,j,k) = stage_xmom(i,j,k) + new_drho_u;                    
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // EB: skip covered y-faces
                if (flag_v(i,j,k).isCovered()) {
                    temp_cur_ymom_arr(i,j,k) = stage_ymom(i,j,k);
                } else {
                    Real new_drho_v = dtau * slow_rhs_rho_v(i,j,k) + dtau * ymom_src_arr(i,j,k);
                    avg_ymom_arr(i,j,k) += facinv * new_drho_v;
                    temp_cur_ymom_arr(i,j,k) = stage_ymom(i,j,k) + new_drho_v;
                }
            });
        } else {
            ParallelFor(tbx, tby,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // EB: skip covered x-faces
                if (flag_u(i,j,k).isCovered()) {

                    temp_cur_xmom_arr(i,j,k) = stage_xmom(i,j,k);

                } else {

                    Real gpx = (theta_extrap(i,j,k) - theta_extrap(i-1,j,k)) * dxi; // Need extrapolation
                    gpx *= mf_ux(i,j,0);

                    Real q = (l_use_moisture) ? myhalf * (qt_arr(i,j,k) + qt_arr(i-1,j,k)) : zero;
                    Real pi_c = myhalf * (pi_stage_ca(i-1,j,k,0) + pi_stage_ca(i,j,k,0));
                    Real fast_rhs_rho_u = -Gamma * R_d * pi_c * gpx / (one + q);

                    Real new_drho_u = prev_xmom(i,j,k) - stage_xmom(i,j,k)
                                    + dtau * fast_rhs_rho_u + dtau * slow_rhs_rho_u(i,j,k)
                                    + dtau * xmom_src_arr(i,j,k);

                    avg_xmom_arr(i,j,k) += facinv * new_drho_u;
                    temp_cur_xmom_arr(i,j,k) = stage_xmom(i,j,k) + new_drho_u;
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // EB: skip covered y-faces
                if (flag_v(i,j,k).isCovered()) {

                    temp_cur_ymom_arr(i,j,k) = stage_ymom(i,j,k);

                } else {

                    Real gpy = (theta_extrap(i,j,k) - theta_extrap(i,j-1,k)) * dyi;
                    gpy *= mf_vy(i,j,0);

                    Real q = (l_use_moisture) ? myhalf * (qt_arr(i,j,k) + qt_arr(i,j-1,k)) : zero;
                    Real pi_c = myhalf * (pi_stage_ca(i,j-1,k,0) + pi_stage_ca(i,j,k,0));
                    Real fast_rhs_rho_v = -Gamma * R_d * pi_c * gpy / (one + q);

                    Real new_drho_v = prev_ymom(i,j,k) - stage_ymom(i,j,k)
                                    + dtau * fast_rhs_rho_v + dtau * slow_rhs_rho_v(i,j,k)
                                    + dtau * ymom_src_arr(i,j,k);

                    avg_ymom_arr(i,j,k) += facinv * new_drho_v;
                    temp_cur_ymom_arr(i,j,k) = stage_ymom(i,j,k) + new_drho_v;
                }
            });
        } // nrk > 0 and/or step > 0
    } //mfi

    // *************************************************************************
    // Main loop: vertical flux / tridiagonal solve / cons update
    // *************************************************************************
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
    std::array<FArrayBox,AMREX_SPACEDIM> flux;
    for ( MFIter mfi(S_stage_data[IntVars::cons],TileNoZ()); mfi.isValid(); ++mfi)
    {
        Box bx  = mfi.tilebox();
        Box tbz = surroundingNodes(bx,2);

        Box vbx = mfi.validbox();
        const auto& vbx_hi = ubound(vbx);

        const Array4<Real const>& zmom_src_arr = zmom_src.const_array(mfi);

        const Array4<const Real>& stage_xmom = S_stage_data[IntVars::xmom].const_array(mfi);
        const Array4<const Real>& stage_ymom = S_stage_data[IntVars::ymom].const_array(mfi);
        const Array4<const Real>& stage_zmom = S_stage_data[IntVars::zmom].const_array(mfi);
        const Array4<const Real>& prim        = S_stage_prim.const_array(mfi);
        const Array4<const Real>& qt_arr      = qt.const_array(mfi);

        const Array4<const Real>& prev_drho_theta = Delta_rho_theta.array(mfi);

        const Array4<const Real>& prev_cons  = S_prev[IntVars::cons].const_array(mfi);
        const Array4<const Real>& stage_cons = S_stage_data[IntVars::cons].const_array(mfi);

        const Array4<const Real>& slow_rhs_cons  = S_slow_rhs[IntVars::cons].const_array(mfi);
        const Array4<const Real>& slow_rhs_rho_w = S_slow_rhs[IntVars::zmom].const_array(mfi);

        const Array4<const Real>& prev_zmom = S_prev[IntVars::zmom].const_array(mfi);
        const Array4<      Real>&  cur_zmom = S_data[IntVars::zmom].array(mfi);

        const Array4<Real>& temp_cur_xmom_arr = temp_cur_xmom.array(mfi);
        const Array4<Real>& temp_cur_ymom_arr = temp_cur_ymom.array(mfi);

        const Array4<Real>& avg_zmom_arr = avg_zmom.array(mfi);

        // Map factors
        const Array4<const Real>& mf_mx = mapfac[MapFacType::m_x]->const_array(mfi);
        const Array4<const Real>& mf_my = mapfac[MapFacType::m_y]->const_array(mfi);
        const Array4<const Real>& mf_uy = mapfac[MapFacType::u_y]->const_array(mfi);
        const Array4<const Real>& mf_vx = mapfac[MapFacType::v_x]->const_array(mfi);

        // EB cell flags
        EBCellFlagFab const& flag_c_fab = (ebfact.get_const_factory())->getMultiEBCellFlagFab()[mfi];
        bool l_singlevalued = (flag_c_fab.getType(bx) == FabType::singlevalued);
        const Array4<const Real> apx_c = l_singlevalued ? (ebfact.get_const_factory())->getAreaFrac()[0]->const_array(mfi) : Array4<const Real>{};
        const Array4<const Real> apy_c = l_singlevalued ? (ebfact.get_const_factory())->getAreaFrac()[1]->const_array(mfi) : Array4<const Real>{};
        const Array4<const Real> apz_c = l_singlevalued ? (ebfact.get_const_factory())->getAreaFrac()[2]->const_array(mfi) : Array4<const Real>{};
        const Array4<const Real> vfrac_c = (ebfact.get_const_factory())->getVolFrac().const_array(mfi);

        // This is MultiCutFab, so

        FArrayBox RHS_fab;  RHS_fab.resize(tbz, 1, The_Async_Arena());
        FArrayBox soln_fab; soln_fab.resize(tbz, 1, The_Async_Arena());

        auto const& RHS_a  = RHS_fab.array();
        auto const& soln_a = soln_fab.array();

        auto const& temp_rhs_arr = temp_rhs.array(mfi);

        auto const&     coeffA_a =     coeff_A_mf.array(mfi);
        auto const& inv_coeffB_a = inv_coeff_B_mf.array(mfi);
        auto const&     coeffC_a =     coeff_C_mf.array(mfi);
        auto const&     coeffP_a =     coeff_P_mf.array(mfi);
        auto const&     coeffQ_a =     coeff_Q_mf.array(mfi);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            flux[dir].resize(surroundingNodes(bx,dir),2,The_Async_Arena());
            flux[dir].setVal<RunOn::Device>(0);
        }
        const GpuArray<const Array4<Real>, AMREX_SPACEDIM>
            flx_arr{{AMREX_D_DECL(flux[0].array(), flux[1].array(), flux[2].array())}};

        // *********************************************************************
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real xflux_lo = (temp_cur_xmom_arr(i  ,j,k) - stage_xmom(i  ,j,k)) / mf_uy(i  ,j,0);
            Real xflux_hi = (temp_cur_xmom_arr(i+1,j,k) - stage_xmom(i+1,j,k)) / mf_uy(i+1,j,0);
            Real yflux_lo = (temp_cur_ymom_arr(i,j  ,k) - stage_ymom(i,j  ,k)) / mf_vx(i,j  ,0);
            Real yflux_hi = (temp_cur_ymom_arr(i,j+1,k) - stage_ymom(i,j+1,k)) / mf_vx(i,j+1,0);

            Real mfsq = mf_mx(i,j,0) * mf_my(i,j,0);

            Real apx_hi = l_singlevalued ? apx_c(i+1,j,k) : one;
            Real apx_lo = l_singlevalued ? apx_c(i  ,j,k) : one;
            Real apy_hi = l_singlevalued ? apy_c(i,j+1,k) : one;
            Real apy_lo = l_singlevalued ? apy_c(i,j  ,k) : one;

            temp_rhs_arr(i,j,k,Rho_comp     ) = ( ( apx_hi * xflux_hi - apx_lo * xflux_lo ) * dxi * mfsq
                                                + ( apy_hi * yflux_hi - apy_lo * yflux_lo ) * dyi * mfsq ) / vfrac_c(i,j,k);

            Real theta_t_x_hi  = ( vfrac_c(i,j,k) * prim(i,j,k,0) + vfrac_c(i+1,j,k) * prim(i+1,j,k,0) ) / (vfrac_c(i,j,k) + vfrac_c(i+1,j,k));
            Real theta_t_x_lo  = ( vfrac_c(i,j,k) * prim(i,j,k,0) + vfrac_c(i-1,j,k) * prim(i-1,j,k,0) ) / (vfrac_c(i,j,k) + vfrac_c(i-1,j,k));
            Real theta_t_y_hi  = ( vfrac_c(i,j,k) * prim(i,j,k,0) + vfrac_c(i,j+1,k) * prim(i,j+1,k,0) ) / (vfrac_c(i,j,k) + vfrac_c(i,j+1,k));
            Real theta_t_y_lo  = ( vfrac_c(i,j,k) * prim(i,j,k,0) + vfrac_c(i,j-1,k) * prim(i,j-1,k,0) ) / (vfrac_c(i,j,k) + vfrac_c(i,j-1,k));

            temp_rhs_arr(i,j,k,RhoTheta_comp) = (( apx_hi * xflux_hi * theta_t_x_hi - apx_lo * xflux_lo * theta_t_x_lo ) * dxi * mfsq +
                                                 ( apy_hi * yflux_hi * theta_t_y_hi - apy_lo * yflux_lo * theta_t_y_lo ) * dyi * mfsq) / vfrac_c(i,j,k);

            if (l_reflux) {
                (flx_arr[0])(i,j,k,0) = xflux_lo;
                (flx_arr[0])(i,j,k,1) = xflux_lo * theta_t_x_lo;
                (flx_arr[1])(i,j,k,0) = yflux_lo;
                (flx_arr[1])(i,j,k,1) = yflux_lo * theta_t_y_lo;
                if (i == vbx_hi.x) {
                    (flx_arr[0])(i+1,j,k,0) = xflux_hi;
                    (flx_arr[0])(i+1,j,k,1) = xflux_hi * theta_t_x_hi;
                }
                if (j == vbx_hi.y) {
                    (flx_arr[1])(i,j+1,k,0) = yflux_hi;
                    (flx_arr[1])(i,j+1,k,1) = yflux_hi * theta_t_y_hi;
                }
            }
        });

        Box bx_shrunk_in_k = bx;
        int klo = tbz.smallEnd(2);
        int khi = tbz.bigEnd(2);
        bx_shrunk_in_k.setSmall(2,klo+1);
        bx_shrunk_in_k.setBig(2,khi-1);

        Real myhalfg = std::abs(myhalf * grav_gpu[2]);
        Real gravity = std::abs(grav_gpu[2]);

        // *********************************************************************
        // Build RHS for vertical tridiagonal solve (interior faces)
        // EB: covered z-faces get RHS = 0 (identity row in make_fast_coeffs
        //     already sets A=0, B=1, C=0 for those faces)
        // *********************************************************************
        ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // EB: covered z-face → leave RHS = 0 (consistent with identity coefficients)
            if (vfrac_w(i,j,k) == zero) {
                RHS_a(i,j,k) = zero;
                return;
            }

            Real q = (l_use_moisture) ? (vfrac_c(i,j,k) * qt_arr(i,j,k) + vfrac_c(i,j,k-1) * qt_arr(i,j,k-1)) / (vfrac_c(i,j,k) + vfrac_c(i,j,k-1)) : zero;

            Real coeff_P = coeffP_a(i,j,k) / (one + q);
            Real coeff_Q = coeffQ_a(i,j,k) / (one + q);

            Real theta_t_lo  = ( vfrac_c(i,j,k-2) * prim(i,j,k-2,PrimTheta_comp) + vfrac_c(i,j,k-1) * prim(i,j,k-1,PrimTheta_comp) ) / (vfrac_c(i,j,k-2) + vfrac_c(i,j,k-1));
            Real theta_t_mid = ( vfrac_c(i,j,k-1) * prim(i,j,k-1,PrimTheta_comp) + vfrac_c(i,j,k  ) * prim(i,j,k  ,PrimTheta_comp) ) / (vfrac_c(i,j,k-1) + vfrac_c(i,j,k  ));
            Real theta_t_hi  = ( vfrac_c(i,j,k  ) * prim(i,j,k  ,PrimTheta_comp) + vfrac_c(i,j,k+1) * prim(i,j,k+1,PrimTheta_comp) ) / (vfrac_c(i,j,k  ) + vfrac_c(i,j,k+1));

            Real Omega_kp1 = prev_zmom(i,j,k+1) - stage_zmom(i,j,k+1);
            Real Omega_k   = prev_zmom(i,j,k  ) - stage_zmom(i,j,k  );
            Real Omega_km1 = prev_zmom(i,j,k-1) - stage_zmom(i,j,k-1);

            Real old_drho_k   = prev_cons(i,j,k  ,Rho_comp) - stage_cons(i,j,k  ,Rho_comp);
            Real old_drho_km1 = prev_cons(i,j,k-1,Rho_comp) - stage_cons(i,j,k-1,Rho_comp);
            Real R0_tmp = coeff_P * prev_drho_theta(i,j,k) + coeff_Q * prev_drho_theta(i,j,k-1)
                        - gravity * ( vfrac_c(i,j,k) * old_drho_k   + vfrac_c(i,j,k-1) * old_drho_km1 ) / (vfrac_c(i,j,k) + vfrac_c(i,j,k-1));

            Real R1_tmp = gravity * ( vfrac_c(i,j,k  ) * (temp_rhs_arr(i,j,k  ,Rho_comp) - slow_rhs_cons(i,j,k  ,Rho_comp)) 
                                    + vfrac_c(i,j,k-1) * (temp_rhs_arr(i,j,k-1,Rho_comp) - slow_rhs_cons(i,j,k-1,Rho_comp)) )
                                    / (vfrac_c(i,j,k) + vfrac_c(i,j,k-1))
                        + coeff_P * (slow_rhs_cons(i,j,k  ,RhoTheta_comp) - temp_rhs_arr(i,j,k  ,RhoTheta_comp)) 
                        + coeff_Q * (slow_rhs_cons(i,j,k-1,RhoTheta_comp) - temp_rhs_arr(i,j,k-1,RhoTheta_comp));

            Real dz_inv = one / dz_ptr[k];
            R1_tmp += beta_1 * dz_inv * ( (Omega_kp1 - Omega_km1)                         * myhalfg
                                         -(Omega_kp1*theta_t_hi  - Omega_k  *theta_t_mid) * coeff_P
                                         -(Omega_k  *theta_t_mid - Omega_km1*theta_t_lo ) * coeff_Q ) / vfrac_w(i,j,k);

            RHS_a(i,j,k) = Omega_k + dtau * (slow_rhs_rho_w(i,j,k) + R0_tmp + dtau * beta_2 * R1_tmp + zmom_src_arr(i,j,k));
        }); // bx_shrunk_in_k

        Box b2d = tbz;
        b2d.setRange(2,0);

        auto const lo = lbound(bx);
        auto const hi = ubound(bx);

        // Boundary faces (domain lo/hi of tile)
        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            // EB: covered domain-boundary z-faces get RHS = 0
            if (vfrac_w(i,j,lo.z) > zero) {
                RHS_a(i,j,lo.z) = prev_zmom(i,j,lo.z) - stage_zmom(i,j,lo.z)
                                 + dtau * slow_rhs_rho_w(i,j,lo.z)
                                 + dtau * zmom_src_arr(i,j,lo.z);
            } else {
                RHS_a(i,j,lo.z) = zero;
            }

            if (vfrac_w(i,j,hi.z+1) > zero) {
                RHS_a(i,j,hi.z+1) = prev_zmom(i,j,hi.z+1) - stage_zmom(i,j,hi.z+1)
                                   + dtau * slow_rhs_rho_w(i,j,hi.z+1)
                                   + dtau * zmom_src_arr(i,j,hi.z+1);
            } else {
                RHS_a(i,j,hi.z+1) = zero;
            }
        }); // b2d

        // *********************************************************************
        // Tridiagonal solve for w (same as NS; identity rows handle covered faces)
        // *********************************************************************
#ifdef AMREX_USE_GPU
        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            soln_a(i,j,lo.z) = RHS_a(i,j,lo.z) * inv_coeffB_a(i,j,lo.z);
            cur_zmom(i,j,lo.z) = stage_zmom(i,j,lo.z) + soln_a(i,j,lo.z);

            for (int k = lo.z+1; k <= hi.z+1; k++) {
                soln_a(i,j,k) = (RHS_a(i,j,k) - coeffA_a(i,j,k)*soln_a(i,j,k-1)) * inv_coeffB_a(i,j,k);
            }

            cur_zmom(i,j,hi.z+1) = stage_zmom(i,j,hi.z+1) + soln_a(i,j,hi.z+1);

            for (int k = hi.z; k >= lo.z; k--) {
                soln_a(i,j,k) -= (coeffC_a(i,j,k) * inv_coeffB_a(i,j,k)) * soln_a(i,j,k+1);
                cur_zmom(i,j,k) = stage_zmom(i,j,k) + soln_a(i,j,k);
            }
        }); // b2d
#else
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                soln_a(i,j,lo.z) = RHS_a(i,j,lo.z) * inv_coeffB_a(i,j,lo.z);
            }
        }
        for (int k = lo.z+1; k <= hi.z+1; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    soln_a(i,j,k) = (RHS_a(i,j,k) - coeffA_a(i,j,k)*soln_a(i,j,k-1)) * inv_coeffB_a(i,j,k);
                }
            }
        }
        for (int j = lo.y; j <= hi.y; ++j) {
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) {
                cur_zmom(i,j,hi.z+1) = stage_zmom(i,j,hi.z+1) + soln_a(i,j,hi.z+1);
            }
        }
        for (int k = hi.z; k >= lo.z; --k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    soln_a(i,j,k) -= (coeffC_a(i,j,k) * inv_coeffB_a(i,j,k)) * soln_a(i,j,k+1);
                    cur_zmom(i,j,k) = stage_zmom(i,j,k) + soln_a(i,j,k);
                }
            }
        }
#endif

        // EB: zero out any remaining non-zero w on fully covered z-faces
        // (should already be zero from identity rows + RHS=0, but enforce for safety)
        ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (vfrac_w(i,j,k) == zero) {
                cur_zmom(i,j,k) = zero;
                soln_a(i,j,k)   = zero;
            }
        });

        if (l_rayleigh_impl_for_w) {
            ParallelFor(bx_shrunk_in_k, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (vfrac_w(i,j,k) > zero) {
                    Real damping_coeff = l_damp_coef * dtau * sinesq_stag_d[k];
                    cur_zmom(i,j,k) /= (one + damping_coeff);
                }
            });
        }

        // *********************************************************************
        // Update rho and (rho theta) from vertical fluxes
        // EB: guarded by vfrac_w for fluxes, vfrac_c for cell update
        // *********************************************************************
        const Array4<Real>& prev_drho_w = Delta_rho_w.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // EB: skip fully covered cells
            if (vfrac_c(i,j,k) == zero) { return; }

            Real zflux_lo = beta_2 * soln_a(i,j,k  ) + beta_1 * prev_drho_w(i,j,k  );
            Real zflux_hi = beta_2 * soln_a(i,j,k+1) + beta_1 * prev_drho_w(i,j,k+1);

            avg_zmom_arr(i,j,k) += facinv * zflux_lo / (mf_mx(i,j,0) * mf_my(i,j,0));
            if (l_reflux) {
                (flx_arr[2])(i,j,k,0) =        zflux_lo / (mf_mx(i,j,0) * mf_my(i,j,0));
                (flx_arr[2])(i,j,k,1) = (flx_arr[2])(i,j,k,0) * myhalf * (prim(i,j,k) + prim(i,j,k-1));
            }

            if (k == vbx_hi.z) {
                avg_zmom_arr(i,j,k+1) += facinv * zflux_hi / (mf_mx(i,j,0) * mf_my(i,j,0));
                if (l_reflux) {
                    (flx_arr[2])(i,j,k+1,0) =          zflux_hi / (mf_mx(i,j,0) * mf_my(i,j,0));
                    (flx_arr[2])(i,j,k+1,1) = (flx_arr[2])(i,j,k+1,0) * myhalf * (prim(i,j,k) + prim(i,j,k+1));
                }
            }

            Real dz_inv = one / dz_ptr[k];
            temp_rhs_arr(i,j,k,Rho_comp     ) += dz_inv * ( zflux_hi - zflux_lo );
            temp_rhs_arr(i,j,k,RhoTheta_comp) += myhalf * dz_inv * ( zflux_hi * (prim(i,j,k) + prim(i,j,k+1))
                                                                     - zflux_lo * (prim(i,j,k) + prim(i,j,k-1)) );
        });

        if (l_reflux) {
            int strt_comp_reflux = 0;
            int  num_comp_reflux = 1;
            if (level < finest_level) {
                fr_as_crse->CrseAdd(mfi,
                    {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                    dx, dtau, strt_comp_reflux, strt_comp_reflux, num_comp_reflux, RunOn::Device);
            }
            if (level > 0) {
                fr_as_fine->FineAdd(mfi,
                    {{AMREX_D_DECL(&(flux[0]), &(flux[1]), &(flux[2]))}},
                    dx, dtau, strt_comp_reflux, strt_comp_reflux, num_comp_reflux, RunOn::Device);
            }
            Gpu::streamSynchronize();
        } // two-way coupling
    } // mfi
    } // OMP

    // *************************************************************************
    // Final: update cons (rho, rho_theta) — EB-guarded
    // *************************************************************************
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(S_stage_data[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<Real>&       cur_cons  = S_data[IntVars::cons].array(mfi);
        const Array4<const Real>& prev_cons = S_prev[IntVars::cons].const_array(mfi);
        auto const& temp_rhs_arr  = temp_rhs.const_array(mfi);
        auto const& slow_rhs_cons = S_slow_rhs[IntVars::cons].const_array(mfi);
        const Array4<Real const>& cc_src_arr = cc_src.const_array(mfi);

        // EB: cell-centered volume fraction
        const Array4<const Real>& vfrac_c = (ebfact.get_const_factory())->getVolFrac().const_array(mfi);

        if (step == 0) {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // EB: skip covered cells
                if (vfrac_c(i,j,k) == zero) { return; }

                cur_cons(i,j,k,Rho_comp)      = prev_cons(i,j,k,Rho_comp) +
                                               dtau * (slow_rhs_cons(i,j,k,Rho_comp) - temp_rhs_arr(i,j,k,Rho_comp));
                cur_cons(i,j,k,RhoTheta_comp) = prev_cons(i,j,k,RhoTheta_comp) +
                                               dtau * (slow_rhs_cons(i,j,k,RhoTheta_comp) - temp_rhs_arr(i,j,k,RhoTheta_comp));
                cur_cons(i,j,k,Rho_comp)      += dtau * cc_src_arr(i,j,k,Rho_comp);
                cur_cons(i,j,k,RhoTheta_comp) += dtau * cc_src_arr(i,j,k,RhoTheta_comp);
            });
        } else {
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // EB: skip covered cells
                if (vfrac_c(i,j,k) == zero) { return; }

                cur_cons(i,j,k,Rho_comp)      += dtau * (slow_rhs_cons(i,j,k,Rho_comp) - temp_rhs_arr(i,j,k,Rho_comp));
                cur_cons(i,j,k,RhoTheta_comp) += dtau * (slow_rhs_cons(i,j,k,RhoTheta_comp) - temp_rhs_arr(i,j,k,RhoTheta_comp));
                cur_cons(i,j,k,Rho_comp)      += dtau * cc_src_arr(i,j,k,Rho_comp);
                cur_cons(i,j,k,RhoTheta_comp) += dtau * cc_src_arr(i,j,k,RhoTheta_comp);
            });
        }

        const Array4<Real>& cur_xmom = S_data[IntVars::xmom].array(mfi);
        const Array4<Real>& cur_ymom = S_data[IntVars::ymom].array(mfi);

        const Array4<Real const>& temp_cur_xmom_arr = temp_cur_xmom.const_array(mfi);
        const Array4<Real const>& temp_cur_ymom_arr = temp_cur_ymom.const_array(mfi);

        // EB: staggered fractions for momentum faces
        const Array4<const Real>& vfrac_u = (ebfact.get_u_const_factory())->getVolFrac().const_array(mfi);
        const Array4<const Real>& vfrac_v = (ebfact.get_v_const_factory())->getVolFrac().const_array(mfi);

        Box tbx = surroundingNodes(bx,0);
        Box tby = surroundingNodes(bx,1);

        ParallelFor(tbx, tby,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // EB: only update open x-faces
            if (vfrac_u(i,j,k) > zero) {
                cur_xmom(i,j,k) = temp_cur_xmom_arr(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // EB: only update open y-faces
            if (vfrac_v(i,j,k) > zero) {
                cur_ymom(i,j,k) = temp_cur_ymom_arr(i,j,k);
            }
        });
    } // mfi
}

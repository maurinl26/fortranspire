! Reference Fortran 90 module — wave_kernels (CPU, no OpenACC pragmas)
!
! Two 2D finite-difference stencils representative of a seismic CPML
! time step: update_vx (velocity update) and update_sigma (stress
! update). Same shape as tests/fixtures/doc_kernel.f90 so the
! transformation pipeline has been exercised against this code shape.
!
! This file is the equivalence baseline — its output is the "golden"
! that the OpenACC variant (openacc.f90) must reproduce within the
! tolerance documented in TOLERANCE.md.

module wave_kernels
  implicit none

contains

  subroutine update_vx(vx, sigma_xx, dx, nx, ny)
    real(8), intent(inout) :: vx(nx, ny)
    real(8), intent(in)    :: sigma_xx(nx, ny)
    real(8), intent(in)    :: dx
    integer, intent(in)    :: nx, ny
    integer :: i, j
    do j = 2, ny
      do i = 2, nx
        vx(i, j) = vx(i, j) + (sigma_xx(i, j) - sigma_xx(i-1, j)) / dx
      end do
    end do
  end subroutine update_vx

  subroutine update_sigma(sigma_xx, vx, vy, dx, dy, nx, ny)
    real(8), intent(inout) :: sigma_xx(nx, ny)
    real(8), intent(in)    :: vx(nx, ny), vy(nx, ny)
    real(8), intent(in)    :: dx, dy
    integer, intent(in)    :: nx, ny
    integer :: i, j
    do j = 2, ny - 1
      do i = 2, nx - 1
        sigma_xx(i, j) = sigma_xx(i, j) &
          + (vx(i+1, j) - vx(i, j)) / dx &
          + (vy(i, j+1) - vy(i, j)) / dy
      end do
    end do
  end subroutine update_sigma

end module wave_kernels

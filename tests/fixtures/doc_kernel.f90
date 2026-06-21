module wave_kernels
  ! Small fixture used by tests/test_document.py and tests/test_analyze.py.
  ! Mimics the structure of a seismic finite-difference stencil with the
  ! classic legacy-Fortran traits (COMMON block, SAVE, implicit typing in
  ! places) so the analyzer and the documenter both have something to chew on.
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

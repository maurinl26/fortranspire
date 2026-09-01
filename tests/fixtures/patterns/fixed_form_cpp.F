C=======================================================================
C Minimal reproduction of the CMAQ dialect: fixed form, column-1
C comments, and live cpp conditionals. The uppercase .F suffix is what
C tells the build system (and us) to run the preprocessor first.
C=======================================================================
      SUBROUTINE ROS3_STEP( N, Y, DT )
      IMPLICIT NONE
      INTEGER N
      REAL*8  Y( N )
      REAL*8  DT
      INTEGER I
#ifdef rbdebug
      WRITE( 6, * ) 'entering ROS3_STEP'
#endif
      DO I = 1, N
         Y( I ) = Y( I ) + DT * MAX( Y( I ), 0.0D0 )
      END DO
      RETURN
      END SUBROUTINE ROS3_STEP

module algor
   implicit  none
   integer, parameter ::  real_dp = selected_int_kind(15)

   private
   public std,moment,median
   public sort
   public histogram, nhist
   public compare_i,compare_s

contains


! compara valor a valor (+=-)
function compare_i(y1,y2) result(v)
   implicit none
   real(real_dp) :: y1(:),y2(:)
   integer :: v(size(y1))

   where(y2==y1)
      v = 2
   elsewhere(y2>y1)
      v = 3
   elsewhere
      v = 1
   end where
end function

! compara series (>,<,=,x)
function compare_s(y1,y2) result(v)
   implicit none
   real(real_dp) :: y1(:),y2(:)
   integer :: v

   if(all(y2>=y1)) then
      if(any(y2>y1)) then
         v = 3  ! domina i
      else
         v = 2  ! iguales
      endif
   elseif(all(y2<=y1)) then
      v = 1  ! domina 1
   else
      v = 4  ! no comparables
   endif
end function


function std(y) result(sdev)
  integer n
  real(real_dp) :: y(:),sdev,var,u
  real(real_dp) :: s(size(y))

  n = size(y)
  u = sum(y)/n
  s = (y-u)

  var=sum(s*s)/(n-1)
  sdev=sqrt(var)

end function


! supone n>1
subroutine moment(data,ave,adev,sdev,var,skew,curt)
  implicit none
  integer :: n,j
  real(real_dp) :: data(:), ave,adev,sdev,var,skew,curt, p,s

  n = size(data)

  s=0.d0
  do j=1,n
    s=s+data(j)
  end do
! calculate mean
  ave=s/n
  adev=0.d0
  var=0.d0
  skew=0.d0
  curt=0.d0
  do j=1,n
    s=data(j)-ave
	adev=adev+dabs(s)
	p=s*s
	var=var+p
	p=p*s
	skew=skew+p
	p=p*s
	skew=skew+p
	curt=curt+p
  end do
  adev=adev/n
  var=var/(n-1)
  sdev=dsqrt(var)
!  if(var.ne.0.d0) then
    skew=skew/(n*sdev**3)
    curt=curt/(n*var**2)-3.d0
!  else
!    print *,' No skew or kurtosis when zero variance.'
!  end if
!  return
end subroutine


SUBROUTINE median(x, n, xmed)

! Find the median of X(1), ... , X(N), using as much of the quicksort
! algorithm as is needed to isolate it.
! N.B. On exit, the array X is partially ordered.

!     Latest revision - 26 November 1996 (de Miller)
IMPLICIT NONE

INTEGER, INTENT(IN)                :: n
REAL(real_dp), INTENT(IN OUT), DIMENSION(:) :: x
REAL(real_dp), INTENT(OUT)                  :: xmed

! Local variables

REAL(real_dp)    :: temp, xhi, xlo, xmax, xmin
LOGICAL :: odd
INTEGER :: hi, lo, nby2, nby2p1, mid, i, j, k

nby2 = n / 2
nby2p1 = nby2 + 1
odd = .true.

!     HI & LO are position limits encompassing the median.

IF (n == 2 * nby2) odd = .false.
lo = 1
hi = n
IF (n < 3) THEN
  IF (n < 1) THEN
    xmed = 0.0
    RETURN
  END IF
  xmed = x(1)
  IF (n == 1) RETURN
  xmed = 0.5*(xmed + x(2))
  RETURN
END IF

!     Find median of 1st, middle & last values.

10 mid = (lo + hi)/2
xmed = x(mid)
xlo = x(lo)
xhi = x(hi)
IF (xhi < xlo) THEN          ! Swap xhi & xlo
  temp = xhi
  xhi = xlo
  xlo = temp
END IF
IF (xmed > xhi) THEN
  xmed = xhi
ELSE IF (xmed < xlo) THEN
  xmed = xlo
END IF

! The basic quicksort algorithm to move all values <= the sort key (XMED)
! to the left-hand end, and all higher values to the other end.

i = lo
j = hi
50 DO
  IF (x(i) >= xmed) EXIT
  i = i + 1
END DO
DO
  IF (x(j) <= xmed) EXIT
  j = j - 1
END DO
IF (i < j) THEN
  temp = x(i)
  x(i) = x(j)
  x(j) = temp
  i = i + 1
  j = j - 1

!     Decide which half the median is in.

  IF (i <= j) GO TO 50
END IF

IF (.NOT. odd) THEN
  IF (j == nby2 .AND. i == nby2p1) GO TO 130
  IF (j < nby2) lo = i
  IF (i > nby2p1) hi = j
  IF (i /= j) GO TO 100
  IF (i == nby2) lo = nby2
  IF (j == nby2p1) hi = nby2p1
ELSE
  IF (j < nby2p1) lo = i
  IF (i > nby2p1) hi = j
  IF (i /= j) GO TO 100

! Test whether median has been isolated.

  IF (i == nby2p1) RETURN
END IF
100 IF (lo < hi - 1) GO TO 10

IF (.NOT. odd) THEN
  xmed = 0.5*(x(nby2) + x(nby2p1))
  RETURN
END IF
temp = x(lo)
IF (temp > x(hi)) THEN
  x(lo) = x(hi)
  x(hi) = temp
END IF
xmed = x(nby2p1)
RETURN

! Special case, N even, J = N/2 & I = J + 1, so the median is
! between the two halves of the series.   Find max. of the first
! half & min. of the second half, then average.

130 xmax = x(1)
DO k = lo, j
  xmax = MAX(xmax, x(k))
END DO
xmin = x(n)
DO k = i, hi
  xmin = MIN(xmin, x(k))
END DO
xmed = 0.5*(xmin + xmax)

RETURN
END SUBROUTINE median


subroutine sort(list)
   implicit none
   REAL(real_dp), DIMENSION (:), INTENT(INOUT)  :: list
   INTEGER :: order(size(list))

   call quick_sort(list, order)
end subroutine


RECURSIVE SUBROUTINE quick_sort(list, order)

! Quick sort routine from:
! Brainerd, W.S., Goldberg, C.H. & Adams, J.C. (1990) "Programmer's Guide to
! Fortran 90", McGraw-Hill  ISBN 0-07-000248-7, pages 149-150.
! Modified by Alan Miller to include an associated integer array which gives
! the positions of the elements in the original order.

IMPLICIT NONE
REAL(real_dp), DIMENSION (:), INTENT(INOUT)  :: list
INTEGER, DIMENSION (:), INTENT(OUT) :: order

! Local variable
INTEGER :: i

DO i = 1, SIZE(list)
  order(i) = i
END DO

CALL quick_sort_1(1, SIZE(list))

CONTAINS

RECURSIVE SUBROUTINE quick_sort_1(left_end, right_end)

INTEGER, INTENT(IN) :: left_end, right_end

!     Local variables
INTEGER             :: i, j, itemp
REAL(real_dp)       :: reference, temp
INTEGER, PARAMETER  :: max_simple_sort_size = 6

IF (right_end < left_end + max_simple_sort_size) THEN
  ! Use interchange sort for small lists
  CALL interchange_sort(left_end, right_end)

ELSE
  ! Use partition ("quick") sort
  reference = list((left_end + right_end)/2)
  i = left_end - 1; j = right_end + 1

  DO
    ! Scan list from left end until element >= reference is found
    DO
      i = i + 1
      IF (list(i) >= reference) EXIT
    END DO
    ! Scan list from right end until element <= reference is found
    DO
      j = j - 1
      IF (list(j) <= reference) EXIT
    END DO


    IF (i < j) THEN
      ! Swap two out-of-order elements
      temp = list(i); list(i) = list(j); list(j) = temp
      itemp = order(i); order(i) = order(j); order(j) = itemp
    ELSE IF (i == j) THEN
      i = i + 1
      EXIT
    ELSE
      EXIT
    END IF
  END DO

  IF (left_end < j) CALL quick_sort_1(left_end, j)
  IF (i < right_end) CALL quick_sort_1(i, right_end)
END IF

END SUBROUTINE quick_sort_1


SUBROUTINE interchange_sort(left_end, right_end)

INTEGER, INTENT(IN) :: left_end, right_end

!     Local variables
INTEGER             :: i, j, itemp
REAL(real_dp)       :: temp

DO i = left_end, right_end - 1
  DO j = i+1, right_end
    IF (list(i) > list(j)) THEN
      temp = list(i); list(i) = list(j); list(j) = temp
      itemp = order(i); order(i) = order(j); order(j) = itemp
    END IF
  END DO
END DO

END SUBROUTINE interchange_sort

END SUBROUTINE quick_sort




subroutine histogram(x,vx,vf,vfa)
!------------------------------------------------------------
! This subroutine takes a real array X(N) and forms a
! plain text 20-bin histogram of the distribution of values.
! Each bin entry output represents SCALE input entries.
!------------------------------------------------------------
  IMPLICIT NONE
  INTEGER :: N
  REAL(real_dp), INTENT(IN) :: X(:)
  real(real_dp) :: vx(:),vf(:),vfa(:)

  REAL(real_dp) :: MIN, MAX, BINWIDTH
  INTEGER :: bins, H(size(vx)), BIN, I

  n = size(x)
  bins = size(vx)

  H=0
  MIN=MINVAL(X)
  MAX=MAXVAL(X)

  BINWIDTH=(MAX-MIN)/REAL(BINS)

  DO I=1,N
    BIN=INT(1+(X(I)-MIN)/BINWIDTH)
    IF ( BIN < 1    ) BIN=1    ! Check for underflows
    IF ( BIN > BINS ) BIN=BINS ! Check for overflows
    H(BIN)=H(BIN)+1
  END DO

  vf = h/real(n,real_dp)
  vx(1)  = min+binwidth/2
  vfa(1) = vf(1)
  do i=2,bins
     vx(i) = min+(i-1)*binwidth+binwidth/2
     vfa(i) = vfa(i-1)+vf(i)
  enddo
  vfa(bins) = 1.0_real_dp ! correccion

end subroutine histogram


! Calcula n
subroutine nhist(n,nb)
   implicit none
   integer :: n,nb

   if(n>=200) then
      nb = max(n/50,20)
   elseif(n>=50) then
      nb = 10
   else
      nb = 5
   endif
end subroutine


end module







#ifdef undef


#endif

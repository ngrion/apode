! Procedimientos basicos
module 	basic
   implicit none
   integer, parameter ::  real_dp = selected_int_kind(15)
   character(len=100) :: disname  ! nombre de la distubucion

   type income
      integer :: n,p
      real(real_dp), allocatable :: y(:,:)
      !resumen
      real(real_dp),allocatable,dimension(:) :: z,min,max,med,ave,adev, &
           sdev, var,skew,kurt
      !Histograma
      integer :: nb
      real(real_dp), allocatable :: vx(:,:),vf(:,:),vfa(:,:)
      !Pobreza y desigualdad
      real(real_dp), allocatable :: pov(:,:),rpg(:,:),tip(:,:)
      integer, allocatable :: q(:),dpov(:,:),dtip(:)
      real(real_dp), allocatable :: ineq(:,:),lorenz(:,:),lorenzg(:,:)
      integer, allocatable :: dineq(:,:),dlorenz(:),dlorenzg(:)

   end type income




   type disp
      logical :: ppg,ppt
      logical :: pdg,pdt
   end type disp

contains


subroutine blank(n)
   implicit none
   integer, optional :: n
   integer :: i,m

   m = 1
   if(present(n)) m = n

   do i=1,m
      write(*,*) ' '
   enddo


end subroutine


subroutine mypause()
    call blank(2)
    write(*,*) '    Presione Enter para continuar... '
    read(*,*)
end subroutine

subroutine print_error(c)
   implicit none
   character(len=*) :: c

   call blank(2)
   write(*,*) '     ERROR: '//trim(c)
   call mypause()

end subroutine


! hacer generica con integer
   pure function int_to_string (i)
      integer,intent(in) :: i
      character(len_int(i)) :: int_to_string
      integer :: temp, d, j

      temp = abs(i)
      do j=len(int_to_string),1,-1
         d = mod(temp,10) + 1
         int_to_string(j:j) = '0123456789'(d:d)
         temp = temp/10
      end do
      if(i < 0) int_to_string(1:1) = '-'

   end function int_to_string

   pure function len_int(i)
      integer, intent(in) :: i
      integer :: len_int
      integer :: temp

      temp = abs(i)
      len_int = 0
      do
         temp = temp/10
         len_int=len_int+1
         if(temp == 0) exit  ! I'd sure like an UNTIL statement
      end do
      if(i < 0) len_int = len_int + 1
   return
   end function len_int



subroutine fecha_string(c)
   implicit none
   character(len=*) c
   character(len=15) mes(12)
   integer :: dt(8)

   mes(1) = 'Enero'
   mes(2) = 'Febrero'
   mes(3) = 'Marzo'
   mes(4) = 'Abril'
   mes(5) = 'Mayo'
   mes(6) = 'Junio'
   mes(7) = 'Julio'
   mes(8) = 'Agosto'
   mes(9) = 'Septiembre'
   mes(10) = 'Octubre'
   mes(11) = 'Noviembre'
   mes(12) = 'Diciembre'

   call date_and_time(values=dt)

   c = int_to_string(dt(3))//' de '//trim(mes(dt(2)))//' de '//&
       int_to_string(dt(1))

end subroutine


end module



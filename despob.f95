module despob
   use basic
   use algor
   implicit none
   real(real_dp) :: uno = 1.0_real_dp

contains

!Los hacea todo iguales: n
subroutine new_income(daty,n,p)
   implicit none
   type(income):: daty
   integer n,p

   daty%p = p
   daty%n = n
   allocate(daty%y(n,p))

end subroutine


subroutine add_income_basic(daty)
   implicit none
   type(income):: daty
   integer p

   p = daty%p

   allocate(daty%z(p))

   allocate(daty%min(p))
   allocate(daty%max(p))
   allocate(daty%med(p))
   allocate(daty%ave(p))
   allocate(daty%adev(p))
   allocate(daty%sdev(p))
   allocate(daty%var(p))
   allocate(daty%skew(p))
   allocate(daty%kurt(p))

end subroutine

subroutine add_income_hist(daty,nb)
   implicit none
   type(income):: daty
   integer nb,p

   p = daty%p
   daty%nb = nb

   if(allocated(daty%vx)) deallocate(daty%vx)
   if(allocated(daty%vf)) deallocate(daty%vf)
   if(allocated(daty%vfa)) deallocate(daty%vfa)

   allocate(daty%vx(nb,p))
   allocate(daty%vf(nb,p))
   allocate(daty%vfa(nb,p))

end subroutine

subroutine add_income_pov(daty,ni)
   implicit none
   type(income):: daty
   integer ni

   if(allocated(daty%pov)) deallocate(daty%pov)
   allocate(daty%pov(ni,daty%p))

   if(allocated(daty%rpg)) deallocate(daty%rpg)
   allocate(daty%rpg(daty%n,daty%p))

   if(allocated(daty%tip)) deallocate(daty%tip)
   allocate(daty%tip(daty%n+1,daty%p))

   if(allocated(daty%q)) deallocate(daty%q)
   allocate(daty%q(daty%p))

   if(allocated(daty%dtip)) deallocate(daty%dtip)
   allocate(daty%dtip(daty%p))

   if(allocated(daty%dpov)) deallocate(daty%dpov)
   allocate(daty%dpov(ni,daty%p))

end subroutine

subroutine add_income_ineq(daty,ni)
   implicit none
   type(income):: daty
   integer ni

   if(allocated(daty%ineq)) deallocate(daty%ineq)
   allocate(daty%ineq(ni,daty%p))

   if(allocated(daty%dineq)) deallocate(daty%dineq)
   allocate(daty%dineq(ni,daty%p))

   if(allocated(daty%lorenz)) deallocate(daty%lorenz)
   allocate(daty%lorenz(daty%n+1,daty%p))

   if(allocated(daty%dlorenz)) deallocate(daty%dlorenz)
   allocate(daty%dlorenz(daty%p))

   if(allocated(daty%lorenzg)) deallocate(daty%lorenzg)
   allocate(daty%lorenzg(daty%n+1,daty%p))

   if(allocated(daty%dlorenzg)) deallocate(daty%dlorenzg)
   allocate(daty%dlorenzg(daty%p))

end subroutine


subroutine clear_income(daty)
   implicit none
   type(income):: daty

   deallocate(daty%y)
   deallocate(daty%z)

   deallocate(daty%min)
   deallocate(daty%max)
   deallocate(daty%med)
   deallocate(daty%ave)
   deallocate(daty%adev)
   deallocate(daty%sdev)
   deallocate(daty%var)
   deallocate(daty%skew)
   deallocate(daty%kurt)

end subroutine

!=======================================================================

! medidas basicas y ordena al ingreso
subroutine basic_measure(daty)
   implicit none
   type(income) :: daty
   real(real_dp) :: med,ave,adev,sdev,var,skew,kurt
   integer :: np,i,n,nb

   np = daty%p
   n  = daty%n
   call nhist(n,nb)
   call add_income_hist(daty,nb)

   do i=1,np
      ! sort
      call sort(daty%y(:,i))
      ! basic  (podrian ser mas eficientes)
      call median(daty%y(:,i), n, med)
      call moment(daty%y(:,i),ave,adev,sdev,var,skew,kurt)
      daty%min(i)  = minval(daty%y(:,i))
      daty%max(i)  = maxval(daty%y(:,i))
      daty%med(i)  = med
      daty%ave(i)  = ave
      daty%adev(i) = adev
      daty%sdev(i) = sdev
      daty%var(i)  = var
      daty%skew(i) = skew
      daty%kurt(i) = kurt
      !hist
      call histogram(daty%y(:,i),daty%vx(:,i),daty%vf(:,i),daty%vfa(:,i))
   enddo

end subroutine


! datos ordenados
subroutine poverty_measure(daty)
   implicit none
   type(income) :: daty
   integer :: ni,i

   ni = 13  !no es seguro!!!!
   call add_income_pov(daty,ni)

   do i=1,daty%p
       call poverty_measure_aux(daty%y(:,i),daty%z(i),daty%rpg(:,i),daty%tip(:,i),daty%q(i),daty%pov(:,i))
       ! comparar indicadores
       daty%dpov(:,i) = compare_i(daty%pov(:,1),daty%pov(:,i))
       ! comparar tip
       daty%dtip(i) = compare_s(daty%tip(:,1),daty%tip(:,i))
   enddo
end subroutine


! ver de chequear size de pov
! calcula rpg,q,pov
subroutine poverty_measure_aux(y,z,rpg,tip,q,pov)
   implicit none
   real(real_dp) :: y(:),z,rpg(:),tip(:),pov(:)
   integer :: q,n,i
   real(real_dp) :: p0,p1,p2,gp,u,a,f,atkp,yedep,ug
   real(real_dp) :: c,k,ep,b  ! elegibles

   n = size(y)

   q = count(y<z)
   rpg(1:q)  = 1-y(1:q)/z
   rpg(q+1:) = 0.0_real_dp

   tip(1) = 0.0_real_dp
   do i=1,n
      tip(i+1) = tip(i)+rpg(i)
   enddo
   tip = tip/n


   ! Indices p0,p1,p2
   p0 = real(q,real_dp)/n
   p1 = sum(rpg(1:q))/n
   p2 = sum(rpg(1:q)**2)/n
   call gini_s(y(1:q),gp)      ! ya ordenado

   ! P0
   pov(1)= p0
   pov(2)= p1
   pov(3)= p2

   !  Indice de Watts (1968)
   pov(4) = sum(log(z/y(1:q)))/n

   ! Indice de Sen
   pov(5) = P0*Gp + P1*(1-Gp)   ! ver si es asintotico

   ! Índice de Sen-Shorrocks-Thon:
   pov(6) = P0*P1*(1+Gp)

   ! Indice de Clark, Ulph y Hemming (1981)
   !c = 0.25   ! c<= 0.5     ! elegible
   !pov(7) =  sum((1-(y(1:q)/z)**c))/(n*c);
   c = 0.5_real_dp  ! 0<=c<=1
   if(c==0) then
      pov(7) =  1 - (product(y(1:q)/z)/n)**(uno/n)
   else
      pov(7) =  1 - ((sum((y(1:q)/z)**c)+(n-q))/n)**(uno/c)
   endif

   ! Indice de Takayama
   u = (sum(y(1:q))+(n-q)*z)/n
   a = 0
   do i=1,q
      a = a+(n-i+1)*y(i)
   enddo
   do i=q+1,n
      a = a+(n-i+1)*z
   enddo
   pov(8) = 1+1/real(n,real_dp) - (2/(u*n*n))*a

   ! Indice de Kakwani
   k = 2  ! elegible
   a = 0
   u = 0
   do i=1,q
       f = (q-i+1)**k
       a = a+f
       u = u+f*(z-y(i))
   enddo
   pov(9) = (q/(n*z*a))*u

   ! Indice de Thon
   u = 0
   do i=1,q
      u = u+(n-i+1)*(z-y(i))
   enddo
   pov(10) = (2/(n*(n+1)*z))*u

   ! Indice de Blackorby y Donaldson
   ep = 2 !elegible
   u = sum(y(1:q))/q
   call atkinson(y(1:q),ep,atkp)
   yedep = u*(1-atkp)
   pov(11) = (real(q,real_dp)/n)*(z-yedep)/z

   !Hagenaars
   !ug = product(y(1:q))**(1.0_real_dp/q)
   ug =  exp(sum(log(y(1:q)))/q) ! o normalizar con el maximo
   pov(12) = (q/real(n,real_dp))*((log(z)-log(ug))/log(z))

   !Chakravarty (1983)
   b = 0.5_real_dp  ! 0<b<1
   pov(13) = sum(uno-((y(1:q)/z)**b))/n




   ! Indice de Zheng (2000) - ver
   !ab = 0.1  ! ab>0
   !Pzhe = sum(exp(ab*(z-yp))-1)/q !  % ver

end subroutine


subroutine inequality_measure(daty)
   implicit none
   type(income) :: daty
   integer :: ni,i

   ni = 20  !no es seguro!!!!
   call add_income_ineq(daty,ni)

   do i=1,daty%p
        call inequality_measure_aux(daty%y(:,i),daty%ave(i),daty%ineq(:,i), &
                          daty%lorenz(:,i),daty%lorenzg(:,i))
       ! comparar indicadores
       daty%dineq(:,i) = compare_i(daty%ineq(:,1),daty%ineq(:,i))
       ! comparar lorenz
       daty%dlorenz(i)  = compare_s(daty%lorenz(:,1),daty%lorenz(:,i))
       daty%dlorenzg(i) = compare_s(daty%lorenzg(:,1),daty%lorenzg(:,i))
   enddo
end subroutine

! Datos ya ordenados
subroutine inequality_measure_aux(y,u,ineq,vlorenz,vlorenzg)
   implicit none
   real(real_dp) :: y(:),ineq(:),u,vlorenz(:),vlorenzg(:),a
   integer :: n,i

   n = size(y)

   ! Lorenz
   vlorenz(1) = 0.0_real_dp
   do i=1,n-1
      vlorenz(i+1) = vlorenz(i)+y(i)
   enddo
   vlorenz = vlorenz/sum(y)
   vlorenz(n+1) = 1.0_real_dp
   vlorenzg = vlorenz*u

   ! Gini
   call gini_s(y,ineq(1)); ! y ordenado

   ! Rango relativo
   !ineq(2) = (maxval(y)-minval(y))/u
   !ineq(2) = (y(n)-y(1))/u
   call ineq_basic(y,ineq(2),'rr')

   ! Desviacion media relativa -Pietra  o Schutz
   !ineq(3) = sum(abs(y-u))/(2*n*u)
   call ineq_basic(y,ineq(3),'dmr')

   ! Coeficiente de variacion
   !ineq(4) = std(y)/u
   call ineq_basic(y,ineq(4),'cv')

   ! Desv Est de los logaritmos
   !ineq(5) = sqrt(sum((log(u)-log(y))**2)/n)
   call ineq_basic(y,ineq(5),'dslog')

   ! General Entropy
    call gentropy(y,0.0_real_dp,ineq(6))
    call gentropy(y,1.0_real_dp,ineq(7))
    call gentropy(y,2.0_real_dp,ineq(8))

   ! Atkinson
    call atkinson(y,0.5_real_dp,ineq(9))
    call atkinson(y,1.0_real_dp,ineq(10))
    call atkinson(y,2.0_real_dp,ineq(11))

    ! Ratios de Kuznets
    call ratios(y,ineq(12:16))

    ! Kolm Family
    a = 2  ! averson a la desigualdad
    call kolm(y,a,ineq(17))

    !Bonferroni
    call bonferroni(y,ineq(18))

    !Lineales
    call ineq_linear(y,ineq(19),'merhan')
    call ineq_linear(y,ineq(20),'piesch')

!    call lorenz(ys)

end subroutine



subroutine ineq_basic(y,m,c)
   implicit none
   real(real_dp) :: y(:),m,u
   integer :: n
   character(len=*) :: c

   n = size(y)
   u = sum(y)/n

   select case(c)
   ! Rango relativo
   case('rr')
       m = (y(n)-y(1))/u !(maxval(y)-minval(y))/u
   ! Desviacion media relativa -Pietra  o Schutz
   case('dmr')
       m = sum(abs(y-u))/(2*n*u)
   ! Coeficiente de variacion
   case('cv')
       m = std(y)/u
   ! Desv Est de los logaritmos
   case('dslog')
       m = sqrt(sum((log(u)-log(y))**2)/n)
   case default
       m = -1
   end select

end subroutine


! indices lineales
subroutine ineq_linear(y,m,c)
   implicit none
   real(real_dp) :: y(:),u,syi,pi,f,qi,m,nn
   integer :: i,n
   character(len=*) :: c

   ! Lorenz
   n   = size(y)
   nn  = real(n,real_dp)
   u   = sum(y)/nn
   syi = y(1)
   pi  = uno/nn
   f   = uno/(nn*u)
   qi  = f*y(1)
   m   = pi-qi

   select case(c)
   case('gini')
      do i=2,n-1
         pi  = i/nn
         syi = syi+y(i)
         qi  = f*syi
         m   = m + (pi-qi)
      enddo
      m = m*2/n
   case('merhan')
      do i=2,n-1
         pi  = i/nn
         syi = syi+y(i)
         qi  = f*syi
         m   = m + (1-pi)*(pi-qi)
      enddo
      m = m*6/n
   case('piesch')
      do i=2,n-1
         pi  = i/nn
         syi = syi+y(i)
         qi  = f*syi
         m   = m + pi*(pi-qi)
      enddo
      m = m*3/n
   case default
      m = -1
   end select

 !  write(*,*) trim(c),'  ',m

end subroutine



subroutine bonferroni(y,r)
   implicit none
   real(real_dp) :: y(:),s,u,x,r
   integer :: i,n

   n = size(y)
   s = y(1)
   x = y(1)
   do i=2,n-1
      x = x+y(i)
      s = s+x/i
   enddo
   u = (x+y(n))/n
   r = 1 - (1/((n-1)*u))*s

end subroutine


subroutine kolm(y,a,r)
   implicit none
   real(real_dp) :: y(:),r,u
   real(real_dp) :: a ! > 0
   integer :: n

   n = size(y)
   u = sum(y)/n
   r = (1/a)*(log((1.0_real_dp/n)*sum(exp(a*(u-y)))))

end subroutine


! Gini (version ya ordenada)
subroutine gini_s(y,g)
   implicit none
   real(real_dp) :: y(:),g
   real(real_dp) :: u,a
   integer :: n,i

   n = size(y)
   u = sum(y)/n
   !call sort(x)

   a = 0
   do i=1,n
      a = a + (n-i+1)*y(i)
   enddo
   g = (n+1)/(n-1) - 2/(n*(n-1)*u)*a
   g = g*(n-1)/n ! ver si esta bien
end subroutine



! Ratios de Kuznets
! Supone datos ordenados
! mejorar resultados para numeros pequeños (se puede imponer rango
! segun valor de n y k)
subroutine ratios(ys,r)
   implicit none
   real(real_dp) :: ys(:),r(:)
   integer :: n,k,i,d

   n = size(ys)
   k = size(r)

   do i=1,k
      d = int(i*(n/(2*real(k))))
      r(i) = sum(ys(n-d+1:n))/sum(ys(1:d))
   enddo
end subroutine


! Entropia general
subroutine gentropy(y,a,p)
   implicit none
   real(real_dp) :: y(:),a,p,u
   integer :: n

   n = size(y)
   u = sum(y)/n

   if(a==0.0) then
      p = sum(log(u/y))/n
   elseif(a==1.0) then
      p = sum((y/u)*log(y/u))/n
   else
      p = (1/(a*(a-1)))*(sum((y/u)**a)/n-1);
   endif
end subroutine

! Atkinson (medida de desigualdad)
subroutine atkinson(y,e,p)
   implicit none
   real(real_dp) :: y(:),e,p,u
   integer :: n

   n = size(y)
   u = sum(y)/n

   if(e==1) then
      !p = 1-sum(log(y)/log(u))/n  ! verificar
      p = 1-product(y**(1.0_real_dp/n))/u
   else
      p = 1-(sum((y/u)**(1-e))/n)**(1.0/(1-e))
   endif
end subroutine


! Supone Ingreso Ordenado
! Hacer opcion para lorenz generalizada
subroutine lorenz(y)
   implicit none
   real(real_dp) :: y(:)
   real(real_dp) :: pa(size(y)+1),ya(size(y)+1)
   integer :: n,i

   n = size(y)

   do i=1,n-1
      pa(i+1) = i
      ya(i+1) = ya(i)+y(i)
   enddo

   pa = pa/real(n,real_dp)
   ya = ya/sum(y)

   pa(1) = 0.0
   ya(1) = 0.0
   pa(n+1) = 1.0
   ya(n+1) = 1.0

!  do i=1,n+1
!      write(*,*) i, pa(i),ya(i)
!  enddo

!   call plot_lorenz_curve(pa,ya)

end subroutine




end module despob






#ifdef undef

 ! Gini
subroutine gini(y,g)
   implicit none
   real(real_dp) :: y(:),g
   real(real_dp) :: x(size(y)),u,a
   integer :: n,i

   n = size(y)
   u = sum(y)/n

   x = y
   call sort(x)

   a = 0
   do i=1,n
      a = a + (n-i+1)*x(i)
   enddo
   g = (n+1)/(n-1) - 2/(n*(n-1)*u)*a
   g = g*(n-1)/n ! ver si esta bien
end subroutine


#endif

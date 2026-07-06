################ Wald  exg stop signal with truncaiton in lower bound of exg

######################## ExGaussian ----

# DISTRIBTION AND RANDOM FUNCTIONS ---- 

# Modified from gamlss.dist to make cdf in tau > 0.05 * sigma case robust,
# and robust to -Inf and Inf inputs, returns NA for bad sigma or tau and
# robust against small sigma cases. Also changes nu name to tau
pexGAUS <- function (q, mu = 5, sigma = 1, tau = 1, lower.tail = TRUE, log.p = FALSE) 
{
    if (sigma <= 0) return(rep(NA,length(q)))
    if (tau <= 0) return(rep(NA,length(q)))
    
    if (sigma < 1e-4) 
      return(pexp(q-mu,1/tau,log=log.p,lower.tail=lower.tail)) # shfited exponential
    
    ly <- length(q)
    sigma <- rep(sigma, length = ly)
    mu <- rep(mu, length = ly)
    tau <- rep(tau, length = ly)
    index <- seq(along = q)
    z <- q - mu - ((sigma^2)/tau)
    cdf <- ifelse(is.finite(q), 
                  ifelse(tau > 0.05 * sigma, 
                         pnorm((q - mu)/sigma) - 
                           exp(log(pnorm(z/sigma)) + ((mu + (sigma^2/tau))^2 - (mu^2) - 
                                                        2 * q * ((sigma^2)/tau))/(2 * sigma^2)), 
                         pnorm(q, mean = mu, sd = sigma)),
                  ifelse(q<0,0,1)   
    )
    if (lower.tail == TRUE) 
      cdf <- cdf
    else cdf <- 1 - cdf
    if (log.p == FALSE) 
      cdf <- cdf
    else cdf <- log(cdf)
    cdf
}
  
# gamlss.dist function, but returns NA for bad sigma or tau, and
# robust against small sigma cases and changed parameter name nu to tau.
dexGAUS <- function (x, mu = 5, sigma = 1, tau = 1, log = FALSE) 
{
    if (sigma <= 0) return(rep(NA,length(x)))
    if (tau <= 0) return(rep(NA,length(x)))
    
    if (sigma < 1e-4) 
      return(dexp(x-mu,1/tau,log=log)) # shfited exponential
    
    ly <- length(x)
    sigma <- rep(sigma, length = ly)
    mu <- rep(mu, length = ly)
    tau <- rep(tau, length = ly)
    z <- x - mu - ((sigma^2)/tau)
    logfy <- ifelse(tau > 0.05 * sigma, -log(tau) - (z + (sigma^2/(2 * 
      tau)))/tau + log(pnorm(z/sigma)), dnorm(x, mean = mu, sd = sigma, 
                                                                                                         log = TRUE))
    if (log == FALSE) 
      fy <- exp(logfy)
    else fy <- logfy
    fy
}
  
# TRUNCATED VERSIONS ----

# n=100; mu=0; sigma=.05; tau=.1; a=0; b=Inf
rEXG <- function (n,mu=.5,sigma=.05,tau=.1,a=-Inf,b=Inf,max.try=1000) 
{
  out <- rnorm(n=n, mean=mu,sd=sigma) + rexp(n=n,rate=1/tau)
  ok <- (out > a) & (out < b)
  if (any(!ok)) {
    nneed <- sum(!ok)
    nget <- ceiling(1.1*nneed/max(c(mean(ok),.001)))
    rtok <- numeric(nneed)
    full <- logical(nneed)
    n.try=0
    repeat {
      tmp <- rnorm(n=nget, mean=mu,sd=sigma) + rexp(n=nget,rate=1/tau) 
      okrt <- (tmp > a) & (tmp < b)
      if ( sum(okrt)>=nneed ) {
        rtok[!full] <- tmp[okrt][1:nneed] 
        break
      } 
      ngot <- sum(okrt)
      if (ngot>0) {
        rtok[!full][1:ngot] <- tmp[okrt]
        full[!full][1:ngot] <- TRUE
      }
      nneed <- nneed - ngot
      if (all(!ok)) 
        nget <- nneed*10 else
          nget <- ceiling(nneed*2/max(c(mean(ok),.001))) 
      n.try <- n.try+1
      if (n.try==max.try) break
    }
    out[!ok] <- rtok
  }
  out
}

dEXG <- function (x, mu = 5, sigma = 1, tau = 1, a=-Inf, b = Inf, log = FALSE) 
  
{
  out <- numeric(length(x))
  ok <- (x>a) | (x<b)
  if (a == -Inf) Fa <- 0 else Fa <- pexGAUS(a,mu=mu,sigma=sigma,tau=tau)
  if (b == Inf) Fb <- 1 else Fb <- pexGAUS(b,mu=mu,sigma=sigma,tau=tau)
  out <- dexGAUS(x[ok],mu,sigma,tau,log=FALSE)/(Fb-Fa)
  if (log) log(out) else out
}
  
pEXG <- function (q, mu = 5, sigma = 1, tau = 1, a=-Inf, b = Inf, 
  lower.tail = TRUE, log.p = FALSE) 
  
{
  out <- numeric(length(q))
  small <- (q<=a); big <- (q>=b); ok <- !small & !big
  out[big] <- 1
  if (a == -Inf) Fa <- 0 else Fa <- pexGAUS(a,mu=mu,sigma=sigma,tau=tau)
  if (b == Inf) Fb <- 1 else Fb <- pexGAUS(b,mu=mu,sigma=sigma,tau=tau)
  out[ok] <- (pexGAUS(q[ok],mu,sigma,tau,log.p=FALSE,lower.tail=lower.tail)-Fa)/(Fb-Fa)
  if (log.p) log(out) else out
}

# par(mfrow=c(1,2))
# n=1e5;mu=.3;sigma=.2;tau=.2;a=0;b=Inf
# sim <- rEXG(n=n,mu=mu,sigma=sigma,tau=tau,a=a,b=b)
# if (!is.finite(b)) hi <- max(sim) else hi <- b
# tmp=hist(sim,breaks=seq(a,hi,length.out=100),freq=FALSE)
# x=tmp$mids; points(x,dEXG(x,mu=mu,sigma=sigma,tau=tau,a=a,b=b),pch=16,cex=.5)
# plot(x,cumsum(tmp$density)/sum(tmp$density),type="l",ylab="cdf")
# points(x,pEXG(x,mu=mu,sigma=sigma,tau=tau,a=a,b=b),pch=16,cex=.5)

######################## Wald ----

# n-choice uniformly varying start point (0-A) Wald 
#    race, with t0, v, A, b (boundary) parameterixaiton

### Single accumulator model ----

# pigt, digt, rwaldt Copyright (C) 2013  Trisha Van Zandt distributed with: 
# Logan, Van Zandt, Verbruggen, and Wagenmakers (2014).  On the ability to 
# inhibit thought and action: General and special theories of an act of control.
# Psychological Review. Comments and changes added by Andrew Heathcote. Trish's
# code is for k = threshold, a = half width of uniform threshold variability,
# l = rate of accumulation. Note that Wald mean = k/l and shape = k^2.

# Following functions use a different parameterization in terms of v=l (rate),
# uniform start point variability from 0-A (A>=0), threshold b (>0) and hence 
# B=b-A (>=0) as a threshold gap. Hence k = b-A/2 = B + A/2 and a=A/2 

rWald <- function(n,B,v,A)
  # random function for single acumulator
{
  
  rwaldt <- function(n,k,l,tiny=1e-6) {
    # random sample of n from a Wald (or Inverse Gaussian)
    # k = criterion, l = rate, assumes sigma=1 Browninan motion
    # about same speed as statmod rinvgauss
    
    rlevy <- function(n=1, m=0, c=1) {
      if (any(c<0)) stop("c must be positive")
      c/qnorm(1-runif(n)/2)^2+m
    }
    
    flag <- l>tiny
    x <- rep(NA,times=n)
    
    x[!flag] <- rlevy(sum(!flag),0,k[!flag]^2)
    mu <- k/l
    lambda <- k^2
    
    y <- rnorm(sum(flag))^2
    mu.0 <- mu[flag]
    lambda.0 <- lambda[flag]
    
    x.0 <- mu.0 + mu.0^2*y/(2*lambda.0) -
      sqrt(4*mu.0*lambda.0*y + mu.0^2*y^2)*mu.0/(2*lambda.0)
    
    z <- runif(length(x.0))
    test <- mu.0/(mu.0+x.0)
    x.0[z>test] <- mu.0[z>test]^2/x.0[z>test]
    x[flag] <- x.0
    x[x<0] <- max(x)
    x
  }
  
  # Act as if negative v never terminates, cluge to do single accumulator
  # case by passing negative v
  if (length(v)!=n) v <- rep(v,length.out=n)
  if (length(B)!=n) B <- rep(B,length.out=n)
  if (length(A)!=n) A <- rep(A,length.out=n)
  
  # Kluge to return -Inf for negative rates, so can implment one accumulator case
  out <- numeric(n)
  ok <- v>0  
  nok <- sum(ok)
  bs <- B[ok]+runif(nok,0,A[ok])
  out[ok] <- rwaldt(nok,k=bs,l=v[ok])
  out[!ok] <- Inf
  out
}


dWald <- function(t,v,B,A)
  # density for single accumulator
{
  
  digt <- function(t,k=1,l=1,a=.1,tiny=1e-10) {
    # pdf of inverse gaussian at t with k +/- a/2 uniform variability
    # returns digt.0 if a<1e-10 
    
    digt.0 <- function(t,k=1,l=1) {
      # pdf of inverse gaussian at t with no k variability
      # much faster than statmod's dinvgauss funciton
      
      lambda <- k^2
      l0 <- l==0
      e <- numeric(length(t))
      if ( any(!l0) ) {
        mu <- k[!l0]/l[!l0]
        e[!l0] <- -(lambda[!l0]/(2*t[!l0])) * (t[!l0]^2/mu^2 - 2*t[!l0]/mu  + 1)
      }
      if ( any(l0) )  e[l0] <- -.5*lambda[l0]/t[l0]
      x <- exp(e + .5*log(lambda) - .5*log(2*t^3*pi))
      x[t<=0] <- 0
      x
    }
    
    options(warn=-1)
    if(length(k)!=length(t)) k <- rep(k,length.out=length(t))
    if(length(l)!=length(t)) l <- rep(l,length.out=length(t))
    if(length(a)!=length(t)) a <- rep(a,length.out=length(t))
    
    tpos <- t<=0
    
    atiny <- a<=tiny & !tpos
    a[atiny] <- 0
    
    ltiny <- (l<=tiny) & !atiny & !tpos
    notltiny <- (l>tiny) & !atiny & !tpos
    l[l<=tiny] <- 0
    
    x <- numeric(length(t))
    
    # No threshold variability
    if ( any(atiny) )
      x[atiny] <- digt.0(t=t[atiny],k=k[atiny],l=l[atiny])
    
    # Threshold variability
    if ( any(!atiny) ) {
      
      if ( any(notltiny) ) { # rate non-zero
        
        sqr.t <- sqrt(t[notltiny])
        
        term.1a <- -(a[notltiny]-k[notltiny]+t[notltiny]*l[notltiny])^2/(2*t[notltiny])
        term.1b <- -(a[notltiny]+k[notltiny]-t[notltiny]*l[notltiny])^2/(2*t[notltiny])
        term.1 <- (exp(term.1a) - exp(term.1b))/sqrt(2*pi*t[notltiny])
        
        term.2a <- log(.5)+log(l[notltiny])
        term.2b <- 2*pnorm((-k[notltiny]+a[notltiny])/sqr.t+sqr.t*l[notltiny])-1
        term.2c <- 2*pnorm((k[notltiny]+a[notltiny])/sqr.t-sqr.t*l[notltiny])-1
        term.2d <- term.2b+term.2c
        term.2 <- exp(term.2a)*term.2d
        
        term.3 <- term.1+term.2
        term.4 <- log(term.3)-log(2)-log(a[notltiny])
        x[notltiny] <- exp(term.4)
      }
      
      if ( any(ltiny) ) {  # rate zero
        log.t <- log(t[ltiny])
        term.1 <- -.5*(log(2)+log(pi)+log.t)
        term.2 <- (k[ltiny]-a[ltiny])^2/(2*t[ltiny])
        term.3 <- (k[ltiny]+a[ltiny])^2/(2*t[ltiny])
        term.4 <- (exp(-term.2)-exp(-term.3))
        term.5 <- term.1+log(term.4) - log(2) - log(a[ltiny])
        x[ltiny] <- exp(term.5)
      }
      
    }
    
    x[x<0 | is.nan(x) ] <- 0
    x
  }
  
  out <- numeric(length(t))
  ok <- v>0
  out[ok] <- digt(t[ok],k=B[ok]+A[ok]/2,l=v[ok],a=A[ok]/2)
  out[!ok] <- 0
  out
}


pWald <- function(t,v,B,A)
  # cumulative density for single accumulator
{
  pigt <- function(t,k=1,l=1,a=.1,tiny=1e-10) {
    # cdf of inverse gaussian at t with k +/- a/2 uniform variability
    # returns pigt.0 if a<=0
    
    pigt.0 <- function(t,k=1,l=1) {
      # cdf of inverse gaussian at t with no k variability
      # much faster than statmod's pinvgauss funciton
      
      mu <- k/l
      lambda <- k^2
      
      e <- exp(log(2*lambda) - log(mu))
      add <- sqrt(lambda/t) * (1 + t/mu)
      sub <- sqrt(lambda/t) * (1 - t/mu)
      
      p.1 <- 1 - pnorm(add)
      p.2 <- 1 - pnorm(sub)
      x <- exp(e + log(p.1)) + p.2
      
      x[t<0] <- 0
      x
    }
    
    options(warn=-1)
    if(length(k)!=length(t)) k <- rep(k,length.out=length(t))
    if(length(l)!=length(t)) l <- rep(l,length.out=length(t))
    if(length(a)!=length(t)) a <- rep(a,length.out=length(t))
    
    tpos <- t<=0
    
    atiny <- a<=tiny & !tpos
    a[atiny] <- 0
    
    ltiny <- (l<=tiny) & !atiny & !tpos
    notltiny <- (l>tiny) & !atiny & !tpos
    l[l<=tiny] <- 0
    
    x <- numeric(length(t))
    
    # No threshold variability
    if ( any(atiny) )
      x[atiny] <- pigt.0(t[atiny],k[atiny],l[atiny])
    
    # Threshold variability
    if ( any(!atiny) ) {
      
      if ( any(notltiny) ) { # rate non-zero
        
        log.t <- log(t[notltiny])
        sqr.t <- sqrt(t[notltiny])
        
        term.1a <- .5*log.t-.5*log(2*pi)
        term.1b <- exp(-((k[notltiny]-a[notltiny]-t[notltiny]*l[notltiny])^2/t[notltiny])/2)
        term.1c <- exp(-((k[notltiny]+a[notltiny]-t[notltiny]*l[notltiny])^2/t[notltiny])/2)
        term.1 <- exp(term.1a)*(term.1b-term.1c)
        
        term.2a <- exp(2*l[notltiny]*(k[notltiny]-a[notltiny]) + 
                         log(pnorm(-(k[notltiny]-a[notltiny]+t[notltiny]*l[notltiny])/sqr.t)))
        term.2b <- exp(2*l[notltiny]*(k[notltiny]+a[notltiny]) + 
                         log(pnorm(-(k[notltiny]+a[notltiny]+t[notltiny]*l[notltiny])/sqr.t)))
        term.2 <- a[notltiny] + (term.2b-term.2a)/(2*l[notltiny])
        
        term.4a <- 2*pnorm((k[notltiny]+a[notltiny])/sqr.t-sqr.t*l[notltiny])-1
        term.4b <- 2*pnorm((k[notltiny]-a[notltiny])/sqr.t-sqr.t*l[notltiny])-1
        term.4c <- .5*(t[notltiny]*l[notltiny] - a[notltiny] - k[notltiny] + .5/l[notltiny])
        term.4d <- .5*(k[notltiny] - a[notltiny] - t[notltiny]*l[notltiny] - .5/l[notltiny])
        term.4 <- term.4c*term.4a + term.4d*term.4b
        
        x[notltiny] <- (term.4 + term.2 + term.1)/(2*a[notltiny])
      }
      
      if ( any(ltiny) ) {  # rate zero
        sqr.t <- sqrt(t[ltiny])
        log.t <- log(t[ltiny])
        term.5a <- 2*pnorm((k[ltiny]+a[ltiny])/sqr.t)-1
        term.5b <- 2*pnorm(-(k[ltiny]-a[ltiny])/sqr.t)-1
        term.5 <- (-(k[ltiny]+a[ltiny])*term.5a - (k[ltiny]-a[ltiny])*term.5b)/(2*a[ltiny])
        
        term.6a <- -.5*(k[ltiny]+a[ltiny])^2/t[ltiny] - .5*log(2) -.5*log(pi) + .5*log.t - log(a[ltiny])
        term.6b <- -.5*(k[ltiny]-a[ltiny])^2/t[ltiny] - .5*log(2) -.5*log(pi) + .5*log.t - log(a[ltiny])
        term.6 <- 1 + exp(term.6b) - exp(term.6a)
        
        x[ltiny] <- term.5 + term.6
      }
      
    }
    
    x[x<0 | is.nan(x) ] <- 0
    x
  }
  
  out <- numeric(length(t))
  ok <- v>0
  out[ok] <- pigt(t[ok],k=B[ok]+A[ok]/2,l=v[ok],a=A[ok]/2)
  out[!ok] <- 0
  out
  
}



######################## Race model ----

rWaldRace <- function(n,v,B,A,t0,return.ttf=FALSE) 
  # random function for Wald race.
{
  B[B<0] <- 0 # Protection for negatives 
  A[A<0] <- 0
  n_v  <- ifelse(is.null(dim(v)), length(v), dim(v)[1])
  ttf <- matrix(t0 + rWald(n*n_v,B=B,v=v,A=A),nrow=n_v)
  if (return.ttf) return(ttf)
  resp <- apply(ttf, 2, which.min)
  data.frame(RT = ttf[cbind(resp,1:n)], R = apply(ttf, 2, which.min))
}

rWaldexgRace <- function(n,v,B,A,t0,gf=0,a=-Inf,b=Inf,return.ttf=FALSE) 
  # random function for Wald race.
{
  B[B<0] <- 0 # Protection for negatives 
  A[A<0] <- 0
  n_v  <- ifelse(is.null(dim(v)), length(v), dim(v)[1])
  ttf <- t0 + rbind(rEXG(n,mu=v[1,1],sigma=B[1,1],tau=A[1,1],a=a[1]),
                    matrix(rWald(n*(n_v-1),B=B[-1,],v=v[-1,],A=A[-1,]),nrow=n_v-1))
  if (return.ttf) return(ttf)
  resp <- apply(ttf, 2, which.min)
  out <- data.frame(RT = ttf[cbind(resp,1:n)], R = apply(ttf, 2, which.min))
  
  if (gf[1] > 0) {
    is.gf <- as.logical(rbinom(dim(out)[1],1,gf))
    out$RT[is.gf] <- NA
    out$R[is.gf] <- 1
  }
  
  out
}

n1Wald <- function(dt,v,B,A,t0=0,gf=0)
  # Generates defective PDF for responses on node=1, dt (decison time) is a vector of times
{
  B[B<0] <- 0 # Protection for negatives 
  A[A<0] <- 0
  n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
  if (is.null(dim(dt))) dt <- matrix(rep(dt,each=n_acc),nrow=n_acc)
  # dt <- dt-t0
  
  is.go <- !is.na(dt[1,])
  n.go <- sum(is.go)
  
  if (!is.matrix(v)) v <- matrix(rep(v,n.go),nrow=n_acc)
  if (!is.matrix(B)) B <- matrix(rep(B,n.go),nrow=n_acc)
  if (!is.matrix(A)) A <- matrix(rep(A,n.go),nrow=n_acc)
  
  # Winner
  dt[1,is.go] <- (1-gf[1])*dWald(dt[1,is.go],A=A[1,],v=v[1,],B=B[1,])
  if (n_acc > 1) for (i in 2:n_acc)
    dt[1,is.go] <- dt[1,is.go]*(1-pWald(dt[i,is.go],A=A[i,],v=v[i,],B=B[i,]))
  
  dt[1,!is.go] <- gf[1]
  
  dt[1,]
}


n1Waldexg <- function(dt,v,B,A,Si=1,t0=0,gf=0,minEXG=-Inf)
  # Generates defective PDF for responses on node=1, dt (decison time) is a vector of times
{
  B[B<0] <- 0 # Protection for negatives 
  A[A<0] <- 0
  n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
  if (is.null(dim(dt))) dt <- matrix(rep(dt,each=n_acc),nrow=n_acc)
  # dt[Si,] <- dt[Si,]-t0
  # dt <- dt-t0
  
  is.go <- !is.na(dt[1,])
  n.go <- sum(is.go)
  
  if (!is.matrix(v)) v <- matrix(rep(v,n.go),nrow=n_acc)
  if (!is.matrix(B)) B <- matrix(rep(B,n.go),nrow=n_acc)
  if (!is.matrix(A)) A <- matrix(rep(A,n.go),nrow=n_acc)
  
  # Winner
  if (Si==1) 
    dt[1,is.go] <- (1-gf[1])*dEXG(dt[1,is.go],mu = v[1,1],sigma = B[1,1],tau = A[1,1],a=minEXG) else
    dt[1,is.go] <- (1-gf[1])*dWald(dt[1,is.go],A=A[1,],v=v[1,],B=B[1,])
  if (n_acc > 1) for (i in 2:n_acc)
    if (Si==i)
      dt[1,is.go] <- dt[1,is.go]*(1-pEXG(q=dt[i,is.go],mu=v[i,1],sigma=B[i,1],tau=A[i,1],a=minEXG)) else
      dt[1,is.go] <- dt[1,is.go]*(1-pWald(dt[i,is.go],A=A[i,],v=v[i,],B=B[i,]))
  
  dt[1,!is.go] <- gf[1]
  
  dt[1,]
}


  # NB1: no st0
  
  # NB2: TRIALS effect on B (threshold), and values < 0 set to 0 
  
rWaldss <- function (n, v, B, t0, A=0, tf=0, gf=0, ts = 0, minEXG=-Inf,
                       SSD=Inf, TRIALS = NA, staircase=NA) 
    # Race among length(v) accumulators (if v a vector), 
    # or dim(v)[1] (if a matrix), first of which is a stop accumulator.
    # Acts the same as rWald except NA returned for RT when winner = 1. 
    # Optional SSD argument can be used to adjust start time for first
    # accumulator. SSD can be a scalar or vector length n; output has an SSD column 
    # For trials with winning first accumulator RT and R set to NA. 
    # tf = trigger failure probability, gf = go failure probability
    # If any !is.na in staircase runs a staircase
    # t0 is a scalar with the standard interpritaiton for GO accumulators. 
    # minEXG is lower bound on EXG wald
  # ts = slope of slowing (speeding if negative) over TRIALS, meanlog - ts*TRIALS
  # This has a linear effect on mean and sd
  
{
    if ( length(SSD)==1 ) SSD <- rep(SSD,n)
    if ( any(is.na(SSD)) || length(SSD) != n )
      stop("SSD cannot have NAs and must be a scalar or same length as n")
    
    n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
    
    # head start for stop accumulator relative to 0 for go accumulators
    t0S <- matrix(rep(c(-t0[1],rep(0,n_acc-1)),length.out=n*n_acc),nrow=n_acc)

    if (!is.matrix(v)) v <- matrix(rep(v,n),nrow=n_acc)
    if (!is.matrix(B)) B <- matrix(rep(B,n),nrow=n_acc)
    if (!is.matrix(A)) A <- matrix(rep(A,n),nrow=n_acc)
    
    if ( !any(is.na(TRIALS)) ) {
      if (length(TRIALS)!=n)
        stop("TRIALS must have length n")
      B[-1,] <- B[-1,] + rep(ts*TRIALS,each=n_acc-1)
    }
    
    if ( gf > 0 ) # Setup for GO failure
      is.gf <- as.logical(rbinom(length(SSD),1,gf)) else
      is.gf <- logical(length(SSD))
    
    if ( all(!is.finite(SSD)) ) {              # ALL GO
      out <- rWaldRace(n,v=v[-1,,drop=FALSE],B=B[-1,,drop=FALSE],A=A[-1,,drop=FALSE],t0=t0)
      out$R <- out$R+1
    } else {                                   # SOME STOP
      if ( any(is.na(staircase)) ) {           # STOP fixed SSD
        # add SSD to stop accumulator
        t0S[1,] <- t0S[1,] + SSD 
        out <- rWaldexgRace(n,v=v,B=B,A=A,t0=t0S,a=minEXG)
        if ( tf>0 ) {
          is.tf <- logical(length(SSD))
          is.tf[is.finite(SSD)][as.logical(rbinom(sum(is.finite(SSD)),1,tf))] <- TRUE  
          if ( any(is.tf) ) { 
            out[is.tf,] <- rWaldRace(sum(is.tf),v=v[-1,is.tf,drop=FALSE],
                                     B=B[-1,is.tf,drop=FALSE],A=A[-1,is.tf,drop=FALSE],t0=0)
            out[is.tf,"R"] <- out[is.tf,"R"]+1
          }
        }
      } else {                                 # STOP, staircase
        if ( !is.numeric(staircase) | length(staircase)!=1 )
          stop("Staircase must be a numeric vector of length 1 specifying the step.")
        SSDi <- SSD[is.finite(SSD)][1] # begining SSD 
        dt <- rWaldexgRace(n,v=v,B=B,A=A,t0=t0S,return.ttf=TRUE,a=minEXG)
        # Setup
        winner <- numeric(n)
        for ( i in c(1:n) ) {
          if ( !is.finite(SSD[i]) )   # not staircase
            dt[1,i] <- dt[1,i] + SSD[i] else
              dt[1,i] <- dt[1,i] + SSDi # staircase
            if ( runif(1)<tf ) # Trigger failure
              winner[i] <- which.min(dt[2:n_acc,i])+1 else
                winner[i] <- which.min(dt[,i])
              if (is.gf[i]) winner[i] <- 1
              if ( is.finite(SSD[i]) ) { # update staircase
                SSD[i] <- SSDi
                if ( winner[i]==1 ) 
                  SSDi <- SSDi + staircase else
                    SSDi <- SSDi - staircase
                  if (SSDi<1e-10) SSDi <- 0
              }
        }
        out <- data.frame(RT=dt[cbind(winner,1:n)],R=winner)
      }
      out$RT <- out$RT + t0 # Add t0 for go responses
    }
    
    # print(out)
    
    out[out$R==1,"RT"] <- NA
    if ( gf > 0 ) {
      out$RT[is.gf] <- NA
      out$R[is.gf] <- 1
    }
    if ( any(is.na(TRIALS)) ) cbind.data.frame(out,SSD=SSD) else
      cbind.data.frame(out,SSD=SSD,TRIALS=TRIALS)
}
  
  
rWaldssABCD <- function (n, v, B, t0,v0,vT,vF,k=1,minEXG=-Inf, 
    A=0, tf=0, gf=0, ts = 0,SSD=Inf, TRIALS = NA, staircase=NA) 
  
    # Like rWaldss except adds change in v with SSD
    # Exponential from v0 to vT and vF

{
    if ( length(SSD)==1 ) SSD <- rep(SSD,n)
    if ( any(is.na(SSD)) || length(SSD) != n )
      stop("SSD cannot have NAs and must be a scalar or same length as n")
    
    n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
    
    # head start for stop accumulator relative to 0 for go accumulators
    t0S <- matrix(rep(c(-t0[1],rep(0,n_acc-1)),length.out=n*n_acc),nrow=n_acc)

    # Get index of true and false rates
    iT <- c(1:length(v))[v==Inf]
    iF <- c(1:length(v))[v==-Inf]
    # Get index of stop-signal trials
    isSS <- is.finite(SSD)

    if (!is.matrix(v)) v <- matrix(rep(v,n),nrow=n_acc)
    if (!is.matrix(B)) B <- matrix(rep(B,n),nrow=n_acc)
    if (!is.matrix(A)) A <- matrix(rep(A,n),nrow=n_acc)
    
    if ( any(!isSS) ) {
      v[iT,!isSS] <- vT  
      v[iF,!isSS] <- vF  
    }
    
    if ( !any(is.na(TRIALS)) ) {
      if (length(TRIALS)!=n)
        stop("TRIALS must have length n")
      B[-1,] <- B[-1,] + rep(ts*TRIALS,each=n_acc-1)
    }
    
    if ( gf > 0 ) # Setup for GO failure
      is.gf <- as.logical(rbinom(length(SSD),1,gf)) else
      is.gf <- logical(length(SSD))
    
    if ( all(!is.finite(SSD)) ) {              # ALL GO
      out <- rWaldRace(n,v=v[-1,,drop=FALSE],B=B[-1,,drop=FALSE],A=A[-1,,drop=FALSE],t0=t0)
      out$R <- out$R+1
    } else {                                   # SOME STOP
      if ( any(is.na(staircase)) ) {           # STOP fixed SSD
        # Set v
        lint <- pmin(k*SSD[isSS],1) 
        v[iT,isSS] <- v0 + (vT-v0)*lint
        v[iF,isSS] <- v0 + (vF-v0)*lint

        
        # add SSD to stop accumulator
        t0S[1,] <- t0S[1,] + SSD 
        out <- rWaldexgRace(n,v=v,B=B,A=A,t0=t0S,a=minEXG)
        if ( tf>0 ) {
          is.tf <- logical(length(SSD))
          is.tf[is.finite(SSD)][as.logical(rbinom(sum(is.finite(SSD)),1,tf))] <- TRUE  
          if ( any(is.tf) ) { 
            out[is.tf,] <- rWaldRace(sum(is.tf),v=v[-1,is.tf,drop=FALSE],
                                     B=B[-1,is.tf,drop=FALSE],A=A[-1,is.tf,drop=FALSE],t0=0)
            out[is.tf,"R"] <- out[is.tf,"R"]+1
          }
        }
      } else {                                 # STOP, staircase
        if ( !is.numeric(staircase) | length(staircase)!=1 )
          stop("Staircase must be a numeric vector of length 1 specifying the step.")
        SSDi <- SSD[is.finite(SSD)][1] # begining SSD 
        dt <- v
        if ( any(!isSS) ) 
          dt[,!isSS] <- rWaldexgRace(n,v=v[,!isSS,drop=FALSE],
            B=B[,!isSS,drop=FALSE],A=A[,!isSS,drop=FALSE],t0=t0S[,!isSS,drop=FALSE],
            return.ttf=TRUE,a=minEXG)
        # Setup
        winner <- numeric(n)
        for ( i in c(1:n) ) {
          if ( !is.finite(SSD[i]) )   # not staircase
            dt[1,i] <- dt[1,i] + SSD[i] else { # staircase
              lint <- min(c(k*SSDi,1))
              v[iT,i] <- v0 + (vT-v0)*lint
              v[iF,i] <- v0 + (vF-v0)*lint
              dt[,i] <- rWaldexgRace(1,v=v[,i,drop=FALSE],B=B[,i,drop=FALSE],
                A=A[,i,drop=FALSE],t0=t0S[,i,drop=FALSE],return.ttf=TRUE,a=minEXG)
              dt[1,i] <- dt[1,i] + SSDi    
            }
          if ( runif(1)<tf ) # Trigger failure
            winner[i] <- which.min(dt[2:n_acc,i])+1 else
              winner[i] <- which.min(dt[,i])
          if (is.gf[i]) winner[i] <- 1
          if ( is.finite(SSD[i]) ) { # update staircase
            SSD[i] <- SSDi
            if ( winner[i]==1 ) 
              SSDi <- SSDi + staircase else
                SSDi <- SSDi - staircase
            if (SSDi<1e-10) SSDi <- 0
          }
        }
        out <- data.frame(RT=dt[cbind(winner,1:n)],R=winner)
      }
      out$RT <- out$RT + t0 # Add t0 for go responses
    }
    
    # print(out)
    
    out[out$R==1,"RT"] <- NA
    if ( gf > 0 ) {
      out$RT[is.gf] <- NA
      out$R[is.gf] <- 1
    }
    if ( any(is.na(TRIALS)) ) cbind.data.frame(out,SSD=SSD) else
      cbind.data.frame(out,SSD=SSD,TRIALS=TRIALS)
}

  
n1PDF.Waldss <- function(rt,v,B,t0,A=0,tf=0,gf=0,ts=0,minEXG=-Inf,
                           SSD=Inf,TRIALS=NA,Si)
    # Same as n1Wald except SSD is either a scalar or vector of length(rt)
    # stop accumulator must have name "NR". SSD is subtracted stop accumulator time
    # and dt=NA done by integration.
    #
    # tf= probabiliy of trigger failure, where
    # L = trigger fail & respond + trigger and respond + trigger and no-response
    #   = tf*L(N-1)+(1-tf)[L(N)+p(S)],
    # L(N-1) = choice race likelihood (no stop accumulator), 
    # L(N) = full N unit race likelihood given they did respond, 
    # p(S) probability of stop winning
    #
  # gf = probabiliy of go failure. 
  # On go trials:   L = go fail (so no response) + go and L above 
  # L = gf + (1-gf)*[tf*L(N-1)+(1-tf)[L(N)+p(S)]]  or similarly
  # L =    [ p(non-response) ]    +           [ p(response) ] 
  #   = [ gf + (1-gf)(1-tf)p(S) ] + [ (1-gf){(tf*Ln(n-1) + (1-tf)*L(N))} ]
  #
  # NB:rt is NOT decision time, but rather full RT as t0 has to be passed
  #    in order to include properly in cases where RT is NA (i.e., sucessful stop)
  
  {
    
    stopfn <- function(t,vj,Bj,Aj,t0,SSD,Si,minEXG) 
    {
      t0S <- rep(0,length(vj)) 
      t0S[-Si] <- t0S[-Si]+SSD
      t0S[Si] <- t0S[Si]+t0
      # t0S[-Si] <- t0S[-Si]-t0   
      # t0S[Si] <- t0S[Si]-SSD
      dt <- matrix(rep(t,each=length(vj)),nrow=length(vj))+t0S
      i <- c(Si,c(1:length(vj))[-Si])
      n1Waldexg(dt[i,,drop=FALSE],v=vj[i],B=Bj[i],A=Aj[i],minEXG=minEXG)
    }
    
    # NOTE: t0 is not subtracted when making dt but passed to handle RT=NA case
    
    if ( length(SSD)==1 ) SSD <- rep(SSD,length(rt))
    if (length(SSD) != length(rt))
      stop("SSD must be a scalar or same length as rt")
    n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
    
    rt <- matrix(rep(rt,each=n_acc),nrow=n_acc)
    is.stop <- is.na(rt[1,])  
    
    if (!is.matrix(v)) v <- matrix(rep(v,dim(rt)[2]),nrow=n_acc)
    if (!is.matrix(B)) B <- matrix(rep(B,dim(rt)[2]),nrow=n_acc)
    if (!is.matrix(A)) A <- matrix(rep(A,dim(rt)[2]),nrow=n_acc)
    
    if ( any(is.na(TRIALS)) | ts == 0 ) { 
      p <- SSD[is.stop]
      pj <- c(1:sum(is.stop))[!duplicated(p)] # index of unique SSD
    } else {
      B[-Si,] <- B[-Si,] + ts*TRIALS
      p <- apply(
        rbind(B[,is.stop,drop=FALSE],SSD[is.stop]),
        2,paste,collapse="")
      pj <- c(1:sum(is.stop))[!duplicated(p)] # index of unique p and SSD
    }
    
    if ( any(!is.stop) ) 
    {
      rt[Si,!is.stop] <- rt[Si,!is.stop] - SSD[!is.stop]
      rt[-Si,!is.stop] <- rt[-Si,!is.stop]-t0
      if ( tf > 0 ) 
      {
        rt[1,!is.stop] <- (1-gf)*(
          tf*n1Wald(rt[-Si,!is.stop,drop=FALSE],v=v[-Si,!is.stop,drop=FALSE],
                    A=A[-Si,!is.stop,drop=FALSE],B=B[-Si,!is.stop,drop=FALSE]) +
            (1-tf)*n1Waldexg(rt[,!is.stop,drop=FALSE],v=v[,!is.stop,drop=FALSE],
                             minEXG=minEXG[1],
                             B=B[,!is.stop,drop=FALSE],A=A[,!is.stop,drop=FALSE],Si=Si)
        )
      } else 
        rt[1,!is.stop] <- (1-gf)*n1Waldexg(dt=rt[,!is.stop,drop=FALSE],
                                        minEXG=minEXG[1],
                                        v=v[,!is.stop,drop=FALSE],B=B[,!is.stop,drop=FALSE],
                                        A=A[,!is.stop,drop=FALSE],Si=Si)
    }
    
    if ( any(is.stop) ) for (j in pj) {
      if ( !is.finite(SSD[j]) ) tmp <- 0 else {
        if (minEXG==-Inf) lower <- -1 else lower <- pmin(minEXG[1]-t0,0)
        tmp <- my.integrate(f=stopfn,lower=minEXG[1]-t0,vj=v[,j],minEXG=minEXG[1],
                            Bj=B[,j],Aj=A[,j],t0=t0,SSD=SSD[j],Si=Si)
      }
      rt[1,is.stop][p %in% p[j]] <- gf +(1-gf)*(1-tf)*tmp
    }
    rt[1,]
}
  

n1PDF.WaldssABCD <- function(rt,v,B,t0,v0,vT,vF,k=1,
  A=0,tf=0,gf=0,ts=0,minEXG=-Inf,SSD=Inf,TRIALS=NA,Si)
    
    # Same as n1PDF.Waldss adds change in v with SSD
    # Exponential from v0 to vT and vF
  
{
    
    stopfn <- function(t,vj,Bj,Aj,t0,SSD,Si,minEXG) 
    {
      t0S <- rep(0,length(vj)) 
      t0S[-Si] <- t0S[-Si]+SSD
      t0S[Si] <- t0S[Si]+t0
      # t0S[-Si] <- t0S[-Si]-t0   
      # t0S[Si] <- t0S[Si]-SSD
      dt <- matrix(rep(t,each=length(vj)),nrow=length(vj))+t0S
      i <- c(Si,c(1:length(vj))[-Si])
      n1Waldexg(dt[i,,drop=FALSE],v=vj[i],B=Bj[i],A=Aj[i],minEXG=minEXG)
    }
    
    # NOTE: t0 is not subtracted when making dt but passed to handle RT=NA case
    
    if ( length(SSD)==1 ) SSD <- rep(SSD,length(rt))
    if (length(SSD) != length(rt))
      stop("SSD must be a scalar or same length as rt")
    n_acc <- ifelse(is.null(dim(v)),length(v),dim(v)[1])
    
    rt <- matrix(rep(rt,each=n_acc),nrow=n_acc)
    is.stop <- is.na(rt[1,])  
    
    # Get index of true and false rates
    iT <- c(1:length(v))[v==Inf]
    iF <- c(1:length(v))[v==-Inf]
    # Get index of stop-signal trials
    isSS <- is.finite(SSD)

    if (!is.matrix(v)) v <- matrix(rep(v,dim(rt)[2]),nrow=n_acc)
    if (!is.matrix(B)) B <- matrix(rep(B,dim(rt)[2]),nrow=n_acc)
    if (!is.matrix(A)) A <- matrix(rep(A,dim(rt)[2]),nrow=n_acc)
    
    if ( any(!isSS) ) {
      v[iT,!isSS] <- vT  
      v[iF,!isSS] <- vF  
    }
    if ( any(isSS) ) {
      lint <- pmin(k*SSD[isSS],1) 
      v[iT,isSS] <- v0 + (vT-v0)*lint
      v[iF,isSS] <- v0 + (vF-v0)*lint
    }

    if ( any(is.na(TRIALS)) | ts == 0 ) { 
      p <- SSD[is.stop]
      pj <- c(1:sum(is.stop))[!duplicated(p)] # index of unique SSD
    } else {
      B[-Si,] <- B[-Si,] + ts*TRIALS
      p <- apply(
        rbind(B[,is.stop,drop=FALSE],SSD[is.stop]),
        2,paste,collapse="")
      pj <- c(1:sum(is.stop))[!duplicated(p)] # index of unique p and SSD
    }
    
    if ( any(!is.stop) ) 
    {
      rt[Si,!is.stop] <- rt[Si,!is.stop] - SSD[!is.stop]
      rt[-Si,!is.stop] <- rt[-Si,!is.stop]-t0
      if ( tf > 0 ) 
      {
        rt[1,!is.stop] <- (1-gf)*(
          tf*n1Wald(rt[-Si,!is.stop,drop=FALSE],v=v[-Si,!is.stop,drop=FALSE],
                    A=A[-Si,!is.stop,drop=FALSE],B=B[-Si,!is.stop,drop=FALSE]) +
            (1-tf)*n1Waldexg(rt[,!is.stop,drop=FALSE],v=v[,!is.stop,drop=FALSE],
                             minEXG=minEXG[1],
                             B=B[,!is.stop,drop=FALSE],A=A[,!is.stop,drop=FALSE],Si=Si)
        )
      } else 
        rt[1,!is.stop] <- (1-gf)*n1Waldexg(dt=rt[,!is.stop,drop=FALSE],
                                        minEXG=minEXG[1],
                                        v=v[,!is.stop,drop=FALSE],B=B[,!is.stop,drop=FALSE],
                                        A=A[,!is.stop,drop=FALSE],Si=Si)
    }
    
    if ( any(is.stop) ) for (j in pj) {
      if ( !is.finite(SSD[j]) ) tmp <- 0 else {
        if (minEXG==-Inf) lower <- -1 else lower <- pmin(minEXG[1]-t0,0)
        tmp <- my.integrate(f=stopfn,lower=minEXG[1]-t0,vj=v[,j],minEXG=minEXG[1],
                            Bj=B[,j],Aj=A[,j],t0=t0,SSD=SSD[j],Si=Si)
      }
      rt[1,is.stop][p %in% p[j]] <- gf +(1-gf)*(1-tf)*tmp
    }
    rt[1,]
}

  
# # VERY EXTENSIVE TESTING WITH Two different SSDs ----
# plot.cell.density <- function(data.cell,C=NA,xlim=NA,limx=c(0,Inf),ymax=NA,lwd=2,
#                               save.density=FALSE,digits=2,main="",show.mean=FALSE,
#             CorrectError.col=c("black","red"),CorrectError.lty=c(1,1))
#   # If !is.na(C) plots density for correct and error responses for a data frame
#   # with columns R (a factor) and RT, adding a boolean score column for R=C.
#   # Otherwise plots each response. Can deal with NA in the RT column, in which
#   # case it provides a summary of p(NA)
# {
#   if (!is.factor(data.cell$R)) data.cell$R <- factor(data.cell$R)
#   if (length(C)==1) C <- rep(C,dim(data.cell)[1])
#   p.na <- mean(is.na(data.cell$RT))
#   is.in <- !is.na(data.cell$RT)
#   is.in[is.in] <- data.cell$RT[is.in]>limx[1] & data.cell$RT[is.in]<limx[2]
#   dat <- data.cell[is.in,]
#   if ( !any(is.na(C)) ) {
#     if ( is.logical(C) & length(C)==dim(data.cell)[1] )
#       dat$C <- C[is.in] else dat$C <- dat$R==C[is.in]
#     if (length(dat$RT[dat$C])>2)
#       dns.correct <- density(dat$RT[dat$C]) else dns.correct <- NULL
#     if (length(dat$RT[!dat$C])>2)
#       dns.error <- density(dat$RT[!dat$C]) else dns.error <- NULL
#     if (is.null(dns.error) & is.null(dns.correct))
#       stop("There are no densities to plot")
#     acc <- mean(dat$C)
#     if (!is.null(dns.correct))
#       dns.correct$y <- dns.correct$y*acc*(1-p.na)
#     if (!is.null(dns.error))
#       dns.error$y <- dns.error$y*(1-acc)*(1-p.na)
#     if (is.na(ymax)) ymax <- max(c(dns.correct$y,dns.error$y))
#     if (!is.null(dns.correct)) {
#       if (any(is.na(xlim))) plot(dns.correct,xlab="RT",ylab="density",ylim=c(0,ymax),main=main,
#         col=CorrectError.col[1],lty=CorrectError.lty[1],lwd=lwd) else
#         plot(dns.correct,xlab="RT",ylab="density",ylim=c(0,ymax),xlim=xlim,main=main,
#           col=CorrectError.col[1],lty=CorrectError.lty[1],lwd=lwd)
#       if (!is.null(dns.error)) lines(dns.error,col=CorrectError.col[2],
#         lty=CorrectError.lty[2],lwd=lwd)
#     } else {
#       if (any(is.na(xlim))) plot(dns.error,xlab="RT",ylab="density",ylim=c(0,ymax),main=main,
#         col=CorrectError.col[2],lty=CorrectError.lty[2],lwd=lwd) else
#         plot(dns.error,xlab="RT",ylab="density",ylim=c(0,ymax),main=main,xlim=xlim,
#           col=CorrectError.col[2],lty=CorrectError.lty[2],lwd=lwd)
#       if (!is.null(dns.correct)) lines(dns.correct,col=CorrectError.col[1],
#         lty=CorrectError.lty[2],lwd=lwd)
#     }
#     nams <- "Accuracy ="
#     ps <- round(acc,digits)
#     if (p.na!=0) {
#       nams <- c(nams,"p(NA) =")
#       ps <- c(ps,round(p.na,digits))
#     }
#     legend("topright",paste(nams,ps),bty="n")
#     legend("topright",xjust=0, inset=c(0,0.1), c("correct","error"),bty="n",
#       lty=CorrectError.lty, col=CorrectError.col,lwd=lwd)
#     if ( save.density ) list(correct=dns.correct,error=dns.error)
#   } else {
#     rs <- levels(dat$R)
#     dns <- vector(mode="list",length=length(rs))
#     names(dns) <- rs
#     ps <- table(dat$R)/dim(dat)[1]
#     for (i in rs) {
#       rt <- dat$RT[dat$R==i]
#       if (length(rt)>2) {
#         dns[[i]] <- density(rt)
#         dns[[i]]$y <- ps[i]*dns[[i]]$y*(1-p.na)
#       }
#     }
#     ymax <- suppressWarnings(max(unlist(lapply(dns,function(x){max(x$y)}))))
#     no.dns <- unlist(lapply(dns,is.null))
#     if (all(no.dns))
#       stop("There are no densities to plot!")
#     dns1 <- dns[!no.dns]
#     ltys <- c(1:length(dns1))
#     plot(dns1[[1]],xlab="RT",ylab="density",ylim=c(0,ymax),lty=ltys[1],
#          main=main)
#     if (length(dns1)>1) for (i in 2:length(dns1)) lines(dns1[[i]],lty=ltys[i])
#     nams <- paste("p(",names(dns1),") =",sep="")
#     ps <- round(ps[!no.dns]*(1-p.na),digits)
#     if ( p.na!=0 ) {
#       nams <- c(nams,"p(NA) =")
#       ps <- c(ps,round(p.na,digits))
#       lty <- c(ltys,NA)
#     }
#     legend("topright",paste(nams,ps),lty=ltys,bty="n",lwd=lwd)
#     if ( save.density ) dns
#   }
# }
# 
# 
# ########### TWO ACCUMULATOR CASE ----
# {
# n=1e4
# # n=10
# v=c(.5,1); B=c(1,1); A=c(1,1); minEXG=0
# SSD = rep(c(1,10)/10,each=n/2)
# 
# # Run one of the follwing two lines
# do.trials=FALSE
# # do.trials = TRUE # requires very differnet plotting check, can be SLOW!
# 
# t0=.2
# 
# ### RUN ONE OF THE FOLLOWING FOUR LINES
# # Without trigger failure or go failure
# tf=0; gf=0
# # With trigger failure, no go failure
# tf=.1;gf=0
# # Without trigger failure, with go failure
# tf=0; gf=.1
# # With trigger failure and go failure
# tf=.1;gf=.1
# 
# if (do.trials) {
#   ts=.5; TRIALS=log10(seq(1,10,length.out=n)) # 1..10 natural so 0-1 on log
#   TRIALS <- as.vector(t(matrix(TRIALS,nrow=2))) # interleave SSDs
#   # Plot slowing in GO (usually nice and linear, up to smooting overfitting)
#   sim.go <- rWaldss(n=n,v=v,B=B,A=A,t0=t0,tf=tf,gf=gf,SSD=SSD,TRIALS=TRIALS,ts=ts,minEXG=minEXG)
#   is.in <- !is.na(sim.go$RT) # in case go failure
#   plot(smooth.spline(TRIALS[is.in],sim.go$RT[is.in]),ylab="Smooth",xlab="TRIALS",type="l")
# } else {TRIALS=NA;ts=0}
# 
# # Simulate stop trials
# sim <- rWaldss(n=n,v=v,B=B,A=A,t0=t0,tf=tf,gf=gf,SSD=SSD,TRIALS=TRIALS,ts=ts,minEXG=minEXG)
# # Sucessful inhibition 
# tapply(is.na(sim$RT),sim$SSD,mean)
# 
# 
# # Plot densities
# par(mfrow=c(1,2))
# dns1 <- plot.cell.density(sim[sim$SSD==.1,],limx=c(0,7),save.density=TRUE,main="SSD=.1")
# dns2 <- plot.cell.density(sim[sim$SSD!=.1,],limx=c(0,7),save.density=TRUE,main="SSD=1")
# x1c <- dns1$'2'$x; x2c <- dns2$'2'$x
# 
# # Signal respond RT
# dat <- sim; dat <- dat[!is.na(dat$RT),]; dat$R <- factor(as.character(dat$R))
# round(tapply(dat$RT,dat[,c("R","SSD")],mean),2)
# 
# if (do.trials) {
#   tmp <- n1PDF.Waldss(sim$RT[!is.na(sim$RT)],v=v[2:1],B=B[2:1],A=A[2:1],t0=t0,minEXG=minEXG,
#     ts=ts,TRIALS=TRIALS[!is.na(sim$RT)],SSD=SSD[!is.na(sim$RT)],Si=2,tf=tf,gf=gf)
#   par(mfrow=c(1,3))
#   # red=black?
#   plot(x1c,dns1$'2'$y,pch=".",main="SSD=.1",ylab="Density",xlab="RT")
#   lines(smooth.spline(sim$RT[!is.na(sim$RT) & SSD==.1],
#    tmp[c(SSD==.1)[!is.na(sim$RT)]]),col="red")
#   # red=black?
#   plot(x2c,dns2$'2'$y,pch=".",main="SSD=1",ylab="Density",xlab="RT")
#   lines(smooth.spline(sim$RT[!is.na(sim$RT) & SSD==1],
#     tmp[c(SSD==1)[!is.na(sim$RT)]]),col="red")
#   # print(tapply(is.na(sim$RT),sim$SSD,mean)) # empirical
#   tmp <- n1PDF.Waldss(rep(NA,n),v=v,B=B,A=A,t0=t0,
#     SSD=SSD,Si=1,tf=tf,gf=gf,ts=ts,TRIALS=TRIALS)
#   # print(mean(tmp[SSD==.1]))
#   # print(mean(tmp[SSD==1]))
#   plot(TRIALS,tmp,pch=".",xlab="TRIALS",ylab="p(NA)",ylim=c(0,1))
#   lines(smooth.spline(TRIALS[SSD==.1],as.numeric(is.na(sim$RT)[SSD==.1])),col="red")
#   lines(smooth.spline(TRIALS[SSD==1],as.numeric(is.na(sim$RT)[SSD==1])),col="red")
# } else {
#   # Save simulated densities
#   r1 <- c(2,1)
#   d.r1 <- n1PDF.Waldss(rt=c(x1c,x2c),v=v[r1],B=B[r1],A=A[r1],t0=t0,minEXG=minEXG,
#     SSD=c(rep(.1,length(x1c)),rep(1,length(x2c))),Si=2,tf=tf,gf=gf)
#   # Plot simulated (black) and theoretical (red) densities
#   par(mfrow=c(1,2))
#   # red=black?
#   plot(x1c,dns1$'2'$y,type="l",main="SSD=.1",ylab="Density",xlab="RT",
#      ylim=c(0,max(dns1$'2'$y)))
#   lines(x1c,d.r1[1:length(x1c)],col="red")
#   # red=black?
#   plot(x2c,dns2$'2'$y,type="l",main="SSD=1",ylab="Density",xlab="RT",
#      ylim=c(0,max(dns2$'2'$y)))
#   lines(x2c,d.r1[(length(x2c)+1):(2*length(x2c))],col="red")
# 
#   # # p(Stop check)
#   # print(tapply(is.na(sim$RT),sim$SSD,mean)) # empirical
#   # print(n1PDF.Waldss(NA,v=v,A=A,B=B,t0=t0,SSD=.1,Si=1,tf=tf,gf=gf,minEXG=minEXG))
#   # print(n1PDF.Waldss(NA,v=v,A=A,B=B,t0=t0,SSD=1,Si=1,tf=tf,gf=gf,minEXG=minEXG))
# }
# 
# }
# 
# ########### THREE ACCUMULATOR CASE ----
# {
# n=1e5
# SSDs = c(1,10)/10
# v=c(1,.75,.25); B=c(1,1,1); A=c(1,1,1); minEXG=0
# SSD = rep(SSDs,each=n/2)
# 
# do.trials=FALSE
# do.trials = TRUE # requires very differnet plotting check, can be SLOW!
# 
# t0=.2
# ### RUN ONE OF THE FOLLOWING FOUR LINES
# # Without trigger failure or go failure
# tf=0; gf=0
# # With trigger failure, no go failure
# tf=.1;gf=0
# # Without trigger failure, with go failure
# tf=0; gf=.1
# # With trigger failure and go failure
# tf=.1;gf=.1
# 
# if (do.trials) {
#   ts=.5; TRIALS=log10(seq(1,10,length.out=n)) # 1..10 natural so 0-1 on log
#   TRIALS <- as.vector(t(matrix(TRIALS,nrow=2))) # interleave SSDs
#   # Plot slowing in GO (usually nice and linear, up to smooting overfitting)
#   sim.go <- rWaldss(n=n,v=v,B=B,A=A,t0=t0,tf=tf,gf=gf,TRIALS=TRIALS,ts=ts,minEXG=minEXG)
#   is.in <- !is.na(sim.go$RT) # in case go failure
#   plot(smooth.spline(TRIALS[is.in],sim.go$RT[is.in]),ylab="Smooth",xlab="TRIALS",type="l")
# } else {TRIALS=NA;ts=0}
# 
# # Simulate stop trials
# par(mfrow=c(1,3)); xlim=c(0,1)
# simgo <- rWaldss(n=n,v=v,B=B,A=A,t0=t0,tf=tf,gf=gf,TRIALS=TRIALS,ts=ts,SSD=Inf,minEXG=minEXG)
# plot.cell.density(simgo,main="Go",xlim=xlim)
# sim <- rWaldss(n=n,v=v,B=B,A=A,t0=t0,tf=tf,gf=gf,TRIALS=TRIALS,ts=ts,SSD=SSD,minEXG=minEXG)
# # Sucessful inhibition
# tapply(is.na(sim$RT),sim$SSD,mean)
# dns1 <- plot.cell.density(sim[sim$SSD==SSDs[1],],limx=c(0,7),save.density=TRUE,main="SSD=.1",xlim=xlim)
# dns2 <- plot.cell.density(sim[sim$SSD!=SSDs[1],],limx=c(0,7),save.density=TRUE,main="SSD=1",xlim=xlim)
# x1c <- dns1$'2'$x; x2c <- dns2$'2'$x
# x1e <- dns1$'3'$x; x2e <- dns2$'3'$x
# 
# # Signal respond RT
# dat <- sim; dat <- dat[!is.na(dat$RT),]; dat$R <- factor(as.character(dat$R))
# round(tapply(dat$RT,dat[,c("R","SSD")],mean),2)
# 
# if (do.trials) {
#   r1 <- c(2,1,3)
#   is.in1 <- !is.na(sim$RT) & sim$R==2
#   d.r1 <- n1PDF.Waldss(sim$RT[is.in1],v=v[r1],ts=ts,TRIALS=TRIALS[is.in1],
#           B=B[r1],A=A[r1],t0=t0,SSD=SSD[is.in1],Si=2,tf=tf,gf=gf,minEXG=minEXG)
#   r2 <- c(3,1,2)
#   is.in2 <- !is.na(sim$RT) & sim$R==3
#   d.r2 <- n1PDF.Waldss(sim$RT[is.in2],v=v[r2],ts=ts,TRIALS=TRIALS[is.in2],
#                    B=B[r2],A=A[r2],t0,SSD=SSD[is.in2],Si=2,tf=tf,gf=gf,minEXG=minEXG)
#   par(mfrow=c(1,3))
#   # red=black?
#   plot(x1c,dns1$'2'$y,pch=".",main="SSD=.1",ylab="Density",xlab="RT",type="l")
#   lines(x1e,dns1$'3'$y,lty=2)
#   lines(smooth.spline(sim$RT[is.in1 & sim$SSD==.1],
#                       d.r1[c(sim$SSD==.1)[is.in1]]),col="red")
#   lines(smooth.spline(sim$RT[is.in2 & sim$SSD==.1],d.r2[c(sim$SSD==.1)[is.in2]]),
#         lty=2,col="red")
#   # red=black?
#   plot(x2c,dns2$'2'$y,pch=".",main="SSD=1",ylab="Density",xlab="RT",type="l")
#   lines(x2e,dns2$'3'$y,lty=2)
#   lines(smooth.spline(sim$RT[is.in1 & sim$SSD==1],
#                       d.r1[c(sim$SSD==1)[is.in1]]),col="red")
#   lines(smooth.spline(sim$RT[is.in2 & sim$SSD==1],
#                       d.r2[c(sim$SSD==1)[is.in2]]),col="red",lty=2)
# 
#   # print(tapply(is.na(sim$RT),sim$SSD,mean)) # empirical
#   tmp <- n1PDF.Waldss(rep(NA,n),v=v,A=A,B=B,t0=t0,SSD=SSD,Si=1,tf=tf,gf=gf,ts=ts,
#     TRIALS=TRIALS,minEXG=minEXG)
#   # print(mean(tmp[SSD==.1]))
#   # print(mean(tmp[SSD==1]))
#   plot(TRIALS,tmp,pch=".",xlab="TRIALS",ylab="p(NA)",ylim=c(0,1))
#   lines(smooth.spline(TRIALS[SSD==.1],as.numeric(is.na(sim$RT)[SSD==.1])),col="red")
#   lines(smooth.spline(TRIALS[SSD==1],as.numeric(is.na(sim$RT)[SSD==1])),col="red")
# } else {
#   # Save simulated densities
#   r1 <- c(2,1,3)
#   d.r1 <- n1PDF.Waldss(rt=c(x1c,x2c),v=v[r1],B=B[r1],A=A[r1],t0=t0,minEXG=minEXG,
#     SSD=c(rep(.1,length(x1c)),rep(1,length(x2c))),Si=2,tf=tf,gf=gf)
#   r2 <- c(3,1,2)
#   d.r2 <- n1PDF.Waldss(rt=c(x1e,x2e),v=v[r2],B=B[r2],A=A[r2],t0=t0,minEXG=minEXG,
#     SSD=c(rep(.1,length(x1e)),rep(1,length(x2e))),Si=2,tf=tf,gf=gf)
#   # Plot simulated (black) and theoretical (red) densities
#   par(mfrow=c(1,2))
#   # red=black?
#   plot(x1c,dns1$'2'$y,type="l",main="SSD=.1",ylab="Density",xlab="RT",
#      ylim=c(0,max(dns1$'2'$y)))
#   lines(x1c,d.r1[1:length(x1c)],col="red")
#   lines(x1e,dns1$'3'$y,lty=2)
#   lines(x1e,d.r2[1:length(x1e)],col="red",lty=2)
#   # red=black?
#   plot(x2c,dns2$'2'$y,type="l",main="SSD=1",ylab="Density",xlab="RT",
#      ylim=c(0,max(dns2$'2'$y)))
#   lines(x2c,d.r1[(length(x2c)+1):(2*length(x2c))],col="red")
#   lines(x2e,dns2$'3'$y,lty=2)
#   lines(x2e,d.r2[(length(x2e)+1):(2*length(x2e))],col="red",lty=2)
# 
#   # # p(Stop check)
#   # print(tapply(is.na(sim$RT),sim$SSD,mean)) # empirical
#   # print(n1PDF.Waldss(NA,v=v,B=B,A=A,t0=t0,SSD=.1,Si=1,tf=tf,gf=gf))
#   # print(n1PDF.Waldss(NA,v=v,B=B,A=A,t0=t0,SSD=1,Si=1,tf=tf,gf=gf))
# }
# }

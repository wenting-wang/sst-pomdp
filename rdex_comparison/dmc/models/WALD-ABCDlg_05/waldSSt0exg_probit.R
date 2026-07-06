# Stop-signal context independent parameterization n-choice Wald racing with an EXG,
# ABCD design with SSD=0 having rate v0 for both accumulators, and changing 
# as a linear growth function to vT and vF for go at rate k. In order to pass things
# appropriate to v.TRUE and v.FALSE v must only vary with M and set constants
# v.TRUE=Inf and v.FALSE=-Inf with factors placed on vT and vF
# External parameters types: "v","B","A","mu","sigma","tau","t0","tf","gf","ts","v0","vT,"vF","k" 
# Internal parameters types: "v","B","A","mu","sigma","tau","t0","tf","gf","ts""v0","vT,"vF","k"
#
# NB1: st0 is not available
# NB2: ts is slope of TRIALS covariate on B (NB: Threshold = B+A)
# 

my.integrate <- function(...,big=10)
# Avoids but in integrate upper=Inf that uses only 1  subdivision
# Use of  big=10 is arbitary ...
{
  out <- try(integrate(...,upper=Inf),silent=TRUE)
  if (class(out)=="try-error") 0 else 
  {
    if (out$subdivisions==1) 
    {
      out <- try(integrate(...,upper=big),silent=TRUE)
      if (class(out)=="try-error") 0 else
      {
         if (out$subdivisions==1) 0 else out$value   
      }
    } else out$value
  }
}

transform.dmc <- function(par.df) 
{
  # Context independence: seperate go and stop accumulator parameterization.
  par.df["NR",c("v","B","A")] <- par.df["NR",c("mu","sigma","tau")]
  par.df$tf <- pnorm(par.df$tf)
  par.df$gf <- pnorm(par.df$gf)
  par.df$k <- par.df$k
  par.df[,c("v","B","A","mu","sigma","tau","t0","tf","gf","ts","v0","vT","vF","k")]
}

random.dmc <- function(n,p.df,model,SSD=Inf,staircase=NULL,TRIALS=NULL)
{
  
  rWaldssABCD(n,v=p.df$v,B=p.df$B,A=p.df$A,t0=p.df$t0[1],
          v0=p.df$v0[1],vT=p.df$vT[1],vF=p.df$vF[1],
          k=p.df$k[1],tf=p.df$tf[1],gf=p.df$gf[1],ts=p.df$ts[1],TRIALS=TRIALS,
          SSD=SSD,staircase=staircase,minEXG = .05)
}

likelihood.dmc <- function(p.vector,data,min.like=1e-10) 
# Returns vector of likelihoods for each RT in data (in same order)
{
  
  likelihood <- numeric(dim(data)[1])
  for ( i in row.names(attr(data,"model")) ) if ( !attr(data,"cell.empty")[i] )
  {
    if ( is.null(data$TRIALS[attr(data,"cell.index")[[i]]]) )
        TRIALS <- NA else TRIALS <- data$TRIALS[attr(data,"cell.index")[[i]]]
    p.df <- p.df.dmc(p.vector,i,attributes(data)$model,n1order=TRUE)
#    if (any(p.df$aT<0) | any(p.df$aF<0)) 
      # likelihood[ attr(data,"cell.index")[[i]] ] <- 
      #   rep(0,length(data$RT[attr(data,"cell.index")[[i]]])) else
      likelihood[ attr(data,"cell.index")[[i]] ] <-
        n1PDF.WaldssABCD(rt=data$RT[attr(data,"cell.index")[[i]]],
          v=p.df$v,
          B=p.df$B,
          A=p.df$A,
          t0=p.df$t0[1], 
          tf=p.df$tf[1],
          gf=p.df$gf[1], 
          ts=p.df$ts[1],
          v0=p.df$v0[1],
          vT=p.df$vT[1],
          vF=p.df$vF[1],
          # aT=p.df$aT[1],
          # aF=p.df$aF[1],
          k=p.df$k[1],
          minEXG = .05,
          # Stop-signal delays
          SSD=data$SSD[attr(data,"cell.index")[[i]]],
          # TRIAL regression
          TRIALS=TRIALS,# In case no TRIALS
          # Index of stop signal accumulator
          Si=c(1:dim(p.df)[1])[row.names(p.df)=="NR"]
      )
 }
 pmax(likelihood,min.like)
}



rm(list=ls())

# 1. Load DMC framework and optimal model (RDEX-ABCD)
source("dmc/dmc.R")
load_model("WALD-ABCDlg_05", "waldSSt0exg_probit.R")

# 2. Define model architecture
model.05 <- model.dmc(
    type="waldss",              
    factors=list(S=c("left","right"), SS=c("GO","SS")),
    responses=c("NR","LEFT","RIGHT"),
    match.map=list(M=list(left="LEFT", right="RIGHT", left="NR")),
    p.map=list(A="1", B="1", v=c("M"), t0="1", gf="1",
               mu="1", sigma="1", tau="1", tf="1", ts="1",
               vT="1", vF="1", v0="1", k="1", minEXG="1"), 
    constants=c(A=0, ts=0, v.true=Inf, v.false=-Inf, minEXG=.05)
)

# 3. Configure literature-informed priors
p.vector <- c(t0=0.137, B=1.417, mu=0.229, sigma=0.076, tau=0.030, 
              v0=2.713, vT=3.285, vF=0.401, gf=0.027, tf=0.037, k=3)
p2 <- c(t0=0.05, B=0.4, mu=0.08, sigma=0.03, tau=0.02, 
        v0=0.5, vT=0.6, vF=0.15, gf=0.01, tf=0.01, k=0.5)

p.prior <- prior.p.dmc(
    dists=rep("tnorm", length(p.vector)), p1=p.vector, p2=p2, 
    lower=c(.1, 0, rep(0,6), rep(NA,2), NA),
    upper=c(1, NA, 2, .5, .5, rep(NA,5), NA)
)

# 4. Read data for 1 subject
subject_id <- "NDAR_INVPWL6YB2C"
input_csv <- paste0(subject_id, "_dmc_ready.csv")
dat <- read.csv(input_csv) 

# Format column properties (DMC strictly requires Factor type)
dat$s <- as.factor(dat$s)
dat$S <- as.factor(dat$S)
dat$SS <- as.factor(dat$SS)
dat$R <- factor(dat$R, levels=c("NR", "LEFT", "RIGHT"))

# Bind data to the model
data.1 <- data.model.dmc(dat, model.05)

# ==========================================
# 5. Fast Test MCMC Sampling Settings
# ==========================================

print("Starting Initial Burn-in...")
sABCD <- h.samples.dmc(nmc=30, p.prior, data.1, thin=2)

print("Running Migration (Unstuck)...")
sABCD <- h.run.unstuck.dmc(sABCD, cores=1, report=5, p.migrate=0.05)

print("Running Convergence Check (Max 1 try)...")
sABCD <- h.run.converge.dmc(
    samples=h.samples.dmc(samples=sABCD, nmc=30, thin=2), 
    nmc=30, cores=1, report=5, verbose=TRUE, max.try=1
)

# ==========================================
# 6. Extract and Save Parameter Estimates to CSV
# ==========================================
print("Quick test finished! Extracting and saving parameters...")

# Extract the samples for the first subject
subj_samples <- sABCD[[1]]

# Calculate the mean of the posterior distributions
estimated_params <- apply(subj_samples$theta, 2, mean)

# Convert the named vector into a 1-row data frame
param_df <- data.frame(t(estimated_params))

# Round values to 4 decimal places for clean export
param_df <- round(param_df, 4)

# Create output filename and save as CSV
output_csv <- paste0(subject_id, "_fitted_params.csv")
write.csv(param_df, file=output_csv, row.names=FALSE)

print(paste("--------------------------------------------------"))
print(paste("Parameters successfully saved to:", output_csv))
print(paste("--------------------------------------------------"))

# Print convergence metric R-hat
gelman.diag.dmc(sABCD)
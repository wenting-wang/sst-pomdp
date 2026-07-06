# fit_model.R
rm(list=ls())
source("dmc/dmc.R")
load_model("WALD-ABCDlg_05", "waldSSt0exg_probit.R")

# 1. Define model architecture
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

# 2. Define literature-informed priors (Weigard et al., 2023)
p.vector <- c(t0=0.137, B=1.417, mu=0.229, sigma=0.076, tau=0.030, 
              v0=2.713, vT=3.285, vF=0.401, gf=0.027, tf=0.037, k=3)
p2 <- c(t0=0.05, B=0.4, mu=0.08, sigma=0.03, tau=0.02, 
        v0=0.5, vT=0.6, vF=0.15, gf=0.01, tf=0.01, k=0.5)

p.prior <- prior.p.dmc(
    dists=rep("tnorm", length(p.vector)), p1=p.vector, p2=p2, 
    lower=c(.1, 0, rep(0,6), rep(NA,2), NA),
    upper=c(1, NA, 2, .5, .5, rep(NA,5), NA)
)

# 3. Load the data prepared by Python
dat.p1 <- read.csv("NDAR_INVPWL6YB2C_dmc_ready.csv")
dat.p2 <- read.csv("NDAR_INVRH6B02GB_dmc_ready.csv")
dat.p3 <- read.csv("NDAR_INVRUVFBX1L_dmc_ready.csv")
okdat.3 <- rbind(dat.p1, dat.p2, dat.p3)

okdat.3$s <- as.factor(okdat.3$s)
okdat.3$S <- as.factor(okdat.3$S)
okdat.3$SS <- as.factor(okdat.3$SS)
okdat.3$R <- factor(okdat.3$R, levels=c("NR", "LEFT", "RIGHT"))

data.3 <- data.model.dmc(okdat.3, model.05)

# ==========================================
# 4. Run DE-MCMC on 3 cores (Full HPC Settings)
# ==========================================
print("Starting Initial Burn-in for 3 subjects...")
sABCD.3 <- h.samples.dmc(nmc=120, p.prior, data.3, thin=10)

print("Running Migration (Unstuck)...")
sABCD.3 <- h.run.unstuck.dmc(sABCD.3, cores=3, report=10, p.migrate=0.05)

print("Running Convergence Check...")
sABCD.3 <- h.run.converge.dmc(
    samples=h.samples.dmc(samples=sABCD.3, nmc=120, thin=10), 
    nmc=40, cores=3, report=10, verbose=TRUE
)

# Save the full posterior chain object for safety and future diagnostics
saveRDS(sABCD.3, file="fitted_3_subjects.RDS")
print("Full posterior samples saved to fitted_3_subjects.RDS")

# ==========================================
# 5. Extract and Save Parameter Estimates to CSV
# ==========================================
print("Extracting parameters for all subjects...")

param_list <- list()

# Loop through each subject in the sample object
for (subj_name in names(sABCD.3)) {
    subj_samples <- sABCD.3[[subj_name]]
    
    # Calculate the mean of the posterior distributions
    estimated_params <- apply(subj_samples$theta, 2, mean)
    
    # Create a 1-row data frame for this subject and prepend the subject ID
    temp_df <- data.frame(subject = subj_name, t(estimated_params))
    param_list[[subj_name]] <- temp_df
}

# Combine all subjects into a single data frame
final_param_df <- do.call(rbind, param_list)

# Round numeric values to 4 decimal places
numeric_cols <- sapply(final_param_df, is.numeric)
final_param_df[, numeric_cols] <- round(final_param_df[, numeric_cols], 4)

# Save the final table to CSV
output_csv <- "fitted_3_subjects_params.csv"
write.csv(final_param_df, file=output_csv, row.names=FALSE)

print("--------------------------------------------------")
print(paste("Parameters successfully extracted and saved to:", output_csv))
print("--------------------------------------------------")

# Print overall Gelman-Rubin convergence metrics
gelman.diag.dmc(sABCD.3)
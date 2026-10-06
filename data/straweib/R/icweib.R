#'Fit stratified Weibull regression model
#'
#'This function fits a stratified Weibull regression model using maximum
#'likelihood estimation. The function can incorporate right, left, interval
#'censored outcomes in addition to fully observed (i.e. uncensored) time to
#'event data. (see details).
#'
#'@param L left endpoint of censoring interval. To indicate
#'left censoring, set L=0.
#'@param R right endpoint of censoring interval. To indicate right
#'censoring, set R=Inf.
#'@param data dataset
#'@param strata variable for stratification. Set it to a character or
#'numeric constant for un-stratified model.
#'@param covariates a formula to specify explanatory variables in the
#'proportional hazards model. The input is a right hand formula object,
#'such as ~x1 + x2. The default is NULL, corresponding to the no covariate case.
#'
#'@details As in the stratified Cox proportional hazards model (Collett (2003)),
#'this model allows a baseline hazard function that is stratum-specific. However,
#'the model assumes that the regression coefficients for all other explanatory
#'variables (excluding the stratum indicator) are constant across strata.
#'Assuming a Weibull distribution for the random variable corresponding to the
#'time to event in conjunction with the Cox proportional hazards model, the survival
#'function can be expressed as S(t | Z) = exp(-lambda*exp(beta*Z)*t^(gamma)), where
#'Z denotes the vector of covariates, gamma denotes the shape parameter and lambda
#'the scale parameter.  To allow stratum-specific baseline hazard functions,
#'we generalize the model given above by  expressing the survival function as
#'S(t | Z, Stratum=i) = exp(-lambda_i*exp(beta*Z)*t^(gamma_i)), where i denotes
#'the stratum, Z denotes the vector of covariates, gamma_i and lambda_i denote
#'the shape and scale parameters for stratum i, respectively.  In particular,
#'the model assumes that the coefficients for explanatory covariates Z (denoted by beta)
#'are the same for all strata i.
#'
#'
#'In the likelihood optimization, u_i=log(lambda_i) and v_i=log(gamma_i) are used as
#'parameters to remove the parameters' range constriction. The likelihood function is
#'optimized using optim() function. The maximum likelihood estimates are used to
#'estimate baseline hazard ratios between two subjects (see \code{\link{HRatio}}),
#'and survival function (see \code{\link{plot.icweib}}).
#'
#'
#'This function can accommondate different types of censored time-to-event outcomes:
#'left censoring, right censoring, interval censoring, and non-censoring (event),
#'by appropriately setting L and R,
#'
#'\tabular{lll}{
#'L     \tab R     \tab INTERPRETATION \cr
#'\cr
#'a   \tab b   \tab interval censoring, [a, b] \cr
#'0   \tab b   \tab left censoring, [0, b] \cr
#'a   \tab Inf \tab right censoring, [a, Inf] \cr
#'a   \tab a   \tab no censoring, event time = a \cr
#'}
#'
#'@return
#'This function returns an object with class \emph{icweib}. The items in the object are,
#'\item{loglik}{log-likelihood functions of the full, reduced, and null models. Reduced model refers to the model that all shape parameters are same. Null model refers to the model that there is no covariate in the model.}
#'\item{coef}{results for estimated coefficients for explanatory variables.}
#'\item{weib}{estimated Weibull shape and scale parameters for each stratum.}
#'\item{stratatest}{results of likelihood ratio test and Wald test corresponding to the null hypothesis that all the strata specific shape parameters are equal.}
#'\item{cov}{covariance matrix of the parameters}
#'\item{ns}{information of different counts}
#'\item{delete}{observation numbers in the data that are deleted due to inappropriate input.}
#'\item{maxT}{maximum observed time in the data}
#'\item{q}{returned object from the \emph{optim} function for the full model.}
#'
#'@seealso \code{\link{HRatio}}, \code{\link{plot.icweib}}
#'
#'@references Collett, D. (2003). \emph{Modelling Survival Data in Medical Research,
#'Second Edition}, Texts in statistical science. Taylor & Francis.
#'
#'@examples
#' ## Analyze tooth data
#' data(tooth24)   ## load data
#' ## Stratified on dmf, and sex as explanatory variable
#' fit <- icweib(L = left, R = right, data = tooth24, strata = dmf, covariates = ~sex)
#'
#' ## Analyze hypernephroma data
#' data(hyper)
#'
#' ## Derive left and right endpoints from time and status
#' hyper$left <- hyper$time
#' hyper$right <- ifelse(hyper$status==1, hyper$time, Inf)
#'
#' ## Stratified on nephrectomy, and age group as explanatory variable
#' fit1 <- icweib(L = left, R = right, data = hyper, strata = nephrectomy, covariates = ~factor(age))
#'
#' @importFrom stats complete.cases model.frame model.matrix na.pass optim
#' pchisq pnorm
#'
#'@export
#'

icweib <-
function(L, R, data, strata="ALL", covariates=NULL) {
  Call <- match.call()
  if (is.name(Call$strata) | is.call(Call$strata)) {
  	straname <- deparse(Call$strata)
  } else {
  	straname <- "strata"
  }
  ## Get strata, left time, right time
  strata <- eval(substitute(strata), data, parent.frame())
  Lt <- eval(substitute(L), data, parent.frame())
  Rt <- eval(substitute(R), data, parent.frame())
  if (length(strata)==1) strata <- rep(strata, length(Lt))
  Lt <- as.vector(Lt)
  Rt <- as.vector(Rt)
  if (!is.numeric(Lt) | !is.numeric(Rt)) stop("L and R must be numeric variables")
  strata <- as.vector(strata)

  ## Model Frame
  if (is.null(covariates)) {
  	mframe <- NULL
  	design <- NULL
  	beta.nm <- NULL
  	nbeta <- 0
  } else {
    mframe <-  model.frame(covariates, data=data, na.action=na.pass)
  }

  ## Compare lengths
  if (length(unique(c(length(Lt), length(Rt), length(strata), dim(mframe)[1])))!=1)
     stop("L, R, strata, and data must have same length")

  ## Remove missing
  nmiss <- complete.cases(cbind(Lt, Rt, strata, mframe))
  validtime <- (Lt >= 0) & (Rt > 0) & (Rt - Lt >= 0)
  validtime[is.na(validtime)] <- F
  keep <- nmiss & validtime
  delete <- which(!keep)
  ndel <- length(delete)
  n <- sum(keep)
  Lt <- Lt[keep]
  Rt <- Rt[keep]
  strata <- strata[keep]
  mframe <- mframe[keep, , drop=F]

  ## Design matrix
  if (!is.null(covariates)) {
  	design <- model.matrix(covariates, data=mframe)[, -1, drop=F]
  	beta.nm <- colnames(design)
    nbeta <- dim(design)[2]
  }

  ## Design matrix for strata
  levels <- sort(unique(strata))
  nstr <- length(levels)
  dstrata <- outer(strata, levels, "==")*1
  stra.nm <- paste(c(rep("v", nstr), rep("u", nstr)), rep(levels, 2), sep=":")

  ## Time information
  allt <- c(Lt, Rt)
  maxT <- max(allt[allt!=Inf])
  meanT <- mean(allt[allt!=0 & allt!=Inf])
  vini <- -log(meanT)

  ## Get event status to see if any event
  et <- Lt==Rt
  event <- sum(et) > 0
  if (event) {
  	ie <- which(et)
  	dstratae <- dstrata[et, ]
  	designe <- design[et, ]
  	Rt[et] <- Inf
  }

  ## Log-likelihood function
  LLt <- log(Lt)
  RRt <- log(Rt)
  loglik <- function(parms) {
    v <- dstrata%*%parms[1:nstr]
    ev <- exp(v)
    u <- dstrata%*%parms[(nstr+1):(2*nstr)]
    if (nbeta > 0) {prog <- design%*%parms[-(1:(2*nstr))]} else {prog <- rep(0, n)}
    lik <- sum(log(exp(-exp(u + prog + ev*LLt)) - exp(-exp(u + prog + ev*RRt))))
    if (event) lik <- lik + sum(u[ie] + prog[ie] + v[ie] + (ev[ie]-1)*LLt[ie])
    return(-lik)
  }

  ## Gradient function
  gradlik <-function(parms) {
    v <- dstrata%*%parms[1:nstr]
    ev <- c(exp(v))
    u <- dstrata%*%parms[(nstr+1):(2*nstr)]
    if (nbeta > 0) {prog <- design%*%parms[-(1:(2*nstr))]} else {prog <- 0}
    SL <- exp(-exp(u + prog + ev*LLt))
    SR <- exp(-exp(u + prog + ev*RRt))
    SLL <- c(SL*exp(u + prog + ev*LLt))
    SRR <- c(ifelse(Rt==Inf, 0, SR*exp(u + prog + ev*RRt)))
    Dev <- colSums((cbind(ifelse(Lt==0, 0, LLt)*ev*dstrata, dstrata, design)*SLL -
           cbind(ifelse(Rt==Inf, 0, RRt)*ev*dstrata, dstrata, design)*SRR)/c(SL-SR))
    if (event) Dev <- Dev - colSums(cbind((1+LLt[ie]*ev[ie])*dstratae, dstratae, designe))
    return(Dev)
  }

  ## Optimization
  parmi <- c(rep(0, nstr), rep(vini, nstr), rep(0, nbeta))
  q <- optim(parmi, loglik, gradlik, method="BFGS", hessian=T)
  if (q$convergence!=0)	warning("Full model not converged")

  ## Covariance matrix
  covm <- solve(q$hessian)

  ## Fit NULL model, no covariates
  loglik0 <- function(parms) {
    v <- dstrata%*%parms[1:nstr]
    ev <- exp(v)
    u <- dstrata%*%parms[(nstr+1):(2*nstr)]
    lik <- sum(log(exp(-exp(u + ev*LLt)) - exp(-exp(u + ev*RRt))))
    if (event) lik <- lik + sum(u[ie] + v[ie] + (ev[ie]-1)*LLt[ie])
    return(-lik)
  }
  gradlik0 <- function(parms) {
    v <- dstrata%*%parms[1:nstr]
    ev <- c(exp(v))
    u <- dstrata%*%parms[(nstr+1):(2*nstr)]
    SL <- exp(-exp(u + ev*LLt))
    SR <- exp(-exp(u + ev*RRt))
    SLL <- c(SL*exp(u + ev*LLt))
    SRR <- c(ifelse(Rt==Inf, 0, SR*exp(u + ev*RRt)))
    Dev <- colSums((cbind(ifelse(Lt==0, 0, LLt)*ev*dstrata, dstrata)*SLL -
           cbind(ifelse(Rt==Inf, 0, RRt)*ev*dstrata, dstrata)*SRR)/c(SL-SR))
    if (event) Dev <- Dev - colSums(cbind((1+LLt[ie]*ev[ie])*dstratae, dstratae))
    return(Dev)
  }
  parmi0 <- c(rep(0, nstr), rep(vini, nstr))
  q0 <- optim(parmi0, loglik0, gradlik0, method="BFGS")

  ## Fit the reduced model: shape parameters are all equal if there are more than 1 strata
  if (nstr > 1) {
    loglik1 <- function(parms) {
      v <- parms[1]
      ev <- exp(v)
      u <- dstrata%*%parms[2:(nstr+1)]
      if (nbeta > 0) {prog <- design%*%parms[-(1:(nstr+1))]} else {prog <- rep(0, n)}
      lik <- sum(log(exp(-exp(u + prog + ev*LLt)) - exp(-exp(u + prog + ev*RRt))))
      if (event) lik <- lik + sum(u[ie] + prog[ie] + v + (ev-1)*LLt[ie])
      return(-lik)
	}
	gradlik1 <-function(parms) {
      v <- parms[1]
      ev <- exp(v)
      u <- dstrata%*%parms[2:(nstr+1)]
      if (nbeta > 0) {prog <- design%*%parms[-(1:(nstr+1))]} else {prog <- 0}
      SL <- exp(-exp(u + prog + ev*LLt))
      SR <- exp(-exp(u + prog + ev*RRt))
      SLL <- c(SL*exp(u + prog + ev*LLt))
      SRR <- c(ifelse(Rt==Inf, 0, SR*exp(u + prog + ev*RRt)))
      Dev <- colSums((cbind(ifelse(Lt==0, 0, LLt)*ev, dstrata, design)*SLL -
              cbind(ifelse(Rt==Inf, 0, RRt)*ev, dstrata, design)*SRR)/c(SL-SR))
      if (event) Dev <- Dev - colSums(cbind(1+LLt[ie]*ev, dstratae, designe))
      return(Dev)
	}
    parmi1 <- c(0, rep(vini, nstr), rep(0, nbeta))
    q1 <- optim(parmi1, loglik1, gradlik1, method="BFGS")
    if (q1$convergence!=0)	warning("Reduced model not converged")

	## Get likelihood ratio test statistic
    df <- nstr - 1
    testlik <- 2*(q1$value - q$value)
    plik <- 1 - pchisq(testlik, df)
    likratio <- data.frame(test="Likelihood Ratio", TestStat=testlik, df, p.value=plik)

	## Wald test statistic
    covv <- covm[1:nstr,1:nstr]
    estv <- q$par[1:nstr]
    testM <- t(sapply(1:(nstr-1), function(x) c(rep(0, x-1), c(1, -1), rep(0, nstr-x-1))))
    testwald <- t(testM%*%estv)%*%solve(testM%*%covv%*%t(testM))%*%(testM%*%estv)
    pwald <- 1 - pchisq(testwald, df)
    tWald <- data.frame(test="Wald", TestStat=testwald, df, p.value=pwald)

	## Combine test statistics
    stratatest <- rbind(tWald, likratio)
    } else {
      stratatest <- NA
      q1 <- list(value=NA)
    }

	## report results
    rownames(covm) <- colnames(covm) <- c(stra.nm, beta.nm)
    logliks <- c(-q$value, -q1$value, -q0$value)
    names(logliks) <- c("full", "reduced", "null")
    if (nbeta > 0) {
	  beta.fit <- q$par[-(1:(2*nstr))]
	  beta.sd <- sqrt(diag(covm)[-(1:(2*nstr))])
	  beta.z <- beta.fit/beta.sd
	  p.value <- 2 * (1 - pnorm(abs(beta.z)))
	  coef <- data.frame(coefficient = beta.fit, SE = beta.sd, z = beta.z, p.value = p.value)
	  rownames(coef) <- beta.nm
    } else {
      coef <- NA
    }
	weib <- data.frame(straname=straname, strata=levels, gamma = exp(q$par[1:nstr]),
	             lambda = exp(q$par[(nstr+1):(2*nstr)]), stringsAsFactors=F)
	ns <- c(n, nstr, nbeta, ndel)
	names(ns) <- c("nused", "nstrata", "ncovariates", "ndeleted")
	z <- list(loglik=logliks, coef=coef, weib=weib, stratatest=stratatest, cov=covm, ns=ns, delete=delete, maxT=maxT, q=q)
	class(z) <- "icweib"
	return(z)
}

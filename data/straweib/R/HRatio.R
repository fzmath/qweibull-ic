#' Estimate hazard ratio between two subjects
#'
#' This function estimates the  hazard ratio between two subjects at given
#' times, along with the associated 95\% confidence interval.
#'
#' @param x results from the model fit returned by the icweib function.
#' @param times the vector of times at which the hazard ratio is estimated.
#' @param NumStra the strata of the subject in the numerator. If the model is
#' unstratified, then set it to be the constant that is used for
#' \emph{strata} argument in \emph{icweib} function, e.g. "ALL".
#' @param NumZ the vector of values of the explanatory variables for
#' the subject in the numerator. The order and length should match
#' the estimated coefficients as shown in x$coef. The default is NULL,
#' corresponding to all 0 or baseline.
#' @param DemStra the strata of the subject in the denominator. If the
#'  model is unstratified, then set it to be the constant that is used
#'  for \emph{strata} argument in \emph{icweib} function, e.g. "ALL".
#' @param DemZ the vector of values of the explanatory variables for the
#' subject in the denominator. The order and length should match the
#' estimated coefficients as shown in x$coef. The default is NULL,
#' corresponding to all 0 or baseline.
#'
#' @details The hazard ratio between two subjects is obtained from the
#' maximum likelihood estimates from the stratified Weibull regression
#' model, along with the corresponding 95\% confidence interval.
#'
#' @return
#' A data frame of estimated hazard ratios and confidence intervals for
#' two subjects at each time point is returned.
#'
#' @seealso \code{\link{icweib}}, \code{\link{plot}}
#'
#' @importFrom stats qnorm
#'
#' @examples
#' data(tooth24)
#' fit <- icweib(L = left, R = right, data = tooth24,
#'               strata = dmf, covariates = ~sex)
#' HRatio(fit, times = 1:7, NumStra = 0, NumZ=0, DemStra = 1, DemZ=0)
#'
#' @export
#'

HRatio <- function(x, times, NumStra, NumZ=NULL, DemStra, DemZ=NULL) {
  nbeta <- x$ns[3]
  if (is.null(NumZ)) NumZ <- rep(0, nbeta)
  if (is.null(DemZ)) DemZ <- rep(0, nbeta)
  if (!is.vector(NumZ)) stop("NumZ must be a vector")
  if (length(NumZ)!=nbeta) stop("Incorrect length for NumZ")
  if (!is.vector(DemZ)) stop("DemZ must be a vector")
  if (length(DemZ)!=nbeta) stop("Incorrect length for DemZ")
  if (nbeta > 0) {
  	dZ <- NumZ - DemZ
  	beta <- x$coef$coefficient
  	prog <- sum(beta*dZ)
  } else {
  	dZ <- NULL
  	prog <- 0
  }

  nstr <- x$ns[2]
  levels <- x$weib$strata
  if (!is.vector(NumStra)) stop("NumStra must be a single value")
  if (length(NumStra) > 1) stop("NumStra must be a single value")
  if (!is.vector(DemStra)) stop("DemStra must be a single value")
  if (length(DemStra) > 1) stop("DemStra must be a single value")

  if (!(NumStra %in% levels)) stop("Invalid value for NumStra")
  if (!(DemStra %in% levels)) stop("Invalid value for DemStra")

  num <- which(levels==NumStra)
  dem <- which(levels==DemStra)

  if (!is.vector(times)) stop("times must be a vector")

  v <- x$q$par[1:nstr]
  u <- x$q$par[(nstr+1):(2*nstr)]
  covm <- x$cov
  logHR <- u[num] + v[num] + log(times)*exp(v[num]) - u[dem] - v[dem] - log(times)*exp(v[dem]) + prog
  getse <- function(tm) {
  	jac1 <- jac2 <- rep(0, 2*nstr)
  	jac1[num] <- 1 + log(tm)*exp(v[num])
  	jac1[nstr + num] <- 1
  	jac2[dem] <- 1 + log(tm)*exp(v[dem])
  	jac2[nstr + dem] <- 1
  	jac <- c(jac1 - jac2, dZ)
  	sqrt(t(jac)%*%covm%*%jac)
  }
  SE <- sapply(times, getse)
  HR <- exp(logHR)
  low95 <- exp(logHR - qnorm(.975)*SE)
  high95 <- exp(logHR + qnorm(.975)*SE)

  HRs <- data.frame(time = times, NumStra, DemStra, prog, HR, low95, high95)
  names(HRs)[4] <- "beta*(Z1-Z2)"
  return(HRs)
}

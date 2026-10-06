#' Plot estimated survival function
#'
#' This function plots the estimated survival function along with associated
#' pointwise 95\% confidence intervals corresponding to the input strata and
#' values of explanatory variables.
#'
#' @param x output returned by icweib function.
#' @param strata the vector of strata for which the survival function is
#' estimated and plotted. The default is NULL, corresponding to all strata.
#' @param Z the vector of values of explanatory variables for which the survival
#' function is estimated and plotted. The order and length should match the
#' estimated coefficients as shown in x$coef. The default is NULL, corresponding
#' to all 0 or baseline.
#' @param tRange the range of time to plot. It should be in the format of c(t1, t2),
#' which means that the range of time is t1 to t2. The default is NULL, corresponding
#' to 0 to maximum observed time in the data.
#' @param tEst the vector of times at which the survival function along with
#' associated pointwise 95\% confidence internval is estimated and output.
#' The default is NULL, which means no estimated survival function is output.
#' @param ... arguments of \code{\link{plot}} function except \emph{col} and
#' \emph{lty}. For example, the axis labels and title of the plot can be specified.
#'
#' @return If tEst is specified, then a dataframe of estimated survival function
#' along with 95\% confidence interval is returned.
#'
#' @details The survival function and associated pointwise 95\% confidence intervals
#' are estimated for input values of time and covariates.
#'
#' @seealso \code{\link{icweib}} \code{\link{plot}}
#'
#' @importFrom graphics legend lines plot
#'
#' @export
#'
#' @examples
#' data(tooth24)
#' fit <- icweib(L = left, R = right, data = tooth24, strata = dmf, covariates = ~sex)
#' surv <- plot(fit, Z = 1, tRange = c(1, 7), tEst=1:7, xlab = "Time", ylab = "Survival Function",
#'              main = "Estimated survival function for sex = 1 (girls)")
#'
plot.icweib <-
function(x, strata=NULL, Z=NULL, tRange=NULL, tEst=NULL, ...) {
  nbeta <- x$ns[3]
  nstr <- x$ns[2]
  levels <- x$weib$strata
  straname <- x$weib$straname[1]

  if (is.null(strata)) strata <- x$weib$strata
  if (is.null(Z)) Z <- rep(0, nbeta)
  if (is.null(tRange)) tRange <- c(0, x$maxT)
  tRange <- sort(tRange)
  if (!(length(tRange)==2 & all(tRange>=0) & all(tRange!=Inf) & (tRange[2] > tRange[1])))
     stop("Invalid input for tRange")
  if (!all(strata %in% levels)) stop("Invalid strata values")
  if (!is.vector(Z)) stop("Z must be a vector")
  if (length(Z)!=nbeta) stop("Incorrect length for Z")

  strata <- sort(unique(strata))
  dstrata <- outer(strata, levels, "==")*1
  parm <- x$q$par
  covm <- x$cov
  estv <- parm[1:nstr]
  estu <- parm[(nstr+1):(2*nstr)]
  estbeta <- parm[-(1:(2*nstr))]
  ts <- seq(tRange[1], tRange[2], length.out=201)
  getstat <- function(vstr, z, t) {
  	if (t==0){
  		return(c(1, 1, 1))
  	}
    surv <- vstr%*%estu + z%*%estbeta + exp(vstr%*%estv)*log(t)
    jac <- c(exp(vstr%*%estv)*log(t)*vstr, vstr, z)
    se <- sqrt(t(jac)%*%covm%*%jac)
    low95 <- surv + qnorm(.975)*se
    high95 <- surv - qnorm(.975)*se
    return(exp(-exp(c(surv, low95, high95))))
  }

  Time <- tRange
  Survival <- c(0, 1)
  plot(Time, Survival, type="n", ...)

  allsurv <- c()
  prog <- if (nbeta>0) Z%*%estbeta else 0
  for (i in (1:length(strata))) {
  	survs <- t(sapply(ts, function(y) getstat(dstrata[i, ], Z, y)))
  	if (!is.null(tEst)) {
  		survest <- t(sapply(tEst, function(y) getstat(dstrata[i, ], Z, y)))
  		allsurv <- rbind(allsurv, data.frame(strata[i], prog, tEst, survest))
  	}
  	lines(ts, survs[, 1], lty=1, col=i, ...)
  	lines(ts, survs[, 2], lty=2, col=i, ...)
  	lines(ts, survs[, 3], lty=2, col=i, ...)
  }
  leg <- paste(straname, strata, sep=": ")
  legend("topright", leg, col=1:length(strata), lty=1)

  if (!is.null(tEst)){
  	  names(allsurv) <- c("strata", "beta*Z", "time", "survival", "low95", "high95")
  	  allsurv
  }
}

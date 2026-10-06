#'Tooth data
#'
#'This data set contains data from the Signal Tandmobiel study, which is
#'described in the paper by Gomez G and others (2009). The time to event
#'is interval censored.
#'
#'@format
#'  A data frame with 4386 observations on the following 5 variables:
#'\describe{
#'  \item{\code{id}}{child id}
#'  \item{\code{left}}{left endpoint of censoring interval.}
#'  \item{\code{right}}{right endpoint of censoring interval.}
#'  \item{\code{sex}}{child's gender. 0 = boy, 1 = girl.}
#'  \item{\code{dmf}}{status of primary predecessor of the tooth.
#'   0 = sound, 1 = decayed or missing due to caries or filled}
#'   }
#'
#'@source \url{http://grass.upc.edu/software/tooth24/copy_of_tooth24-data-set/view}
#'
#'@references G. Gomez, M. Calle, R. Oller, and K. Langohr (2009). Tutorial on methods
#'for interval-censored data and their implementation in R.
#'\emph{Statistical Modeling} 9(4), 259
#'
#'@examples
#'data(tooth24)
"tooth24"


#'Simulated data with mixed types of events
#'
#'This simulated data contains event times that are left censored, right
#'censored, interval censored, or non-censored (observed event). The data is
#'generated from a stratified Weibull distribution model in which each stratum
#'is assumed to have an independent stratum-specific shape parameter. In
#'addition, the regression coefficients corresponding to the vector of
#'explanatory variables excluding the stratum indicator are assumed to be
#'constant across strata.
#'
#'@format   A data frame with 298 observations on the following 6 variables:
#'  \describe{ \item{\code{ID}}{subject id} \item{\code{strata}}{strata}
#'  \item{\code{cov1}}{a continuous covariate} \item{\code{cov2}}{a continuous
#'  covariate} \item{\code{left}}{left endpoint of censoring interval}
#'  \item{\code{right}}{right endpoint of censoring interval} }
#'
#'@references see \code{\link{icweib}} for details on how to set \emph{L} and
#'  \emph{R} for different types of events.
#'
#'@examples
#'data(simdata)
#'
"simdata"



#'Treatment of hypernephroma data 
#'
#'This dataset contains survival times for 36
#'patients with malignant tumour in the kidney. Some of the patients received
#'nephrectomy. See Example 3.4 and example 5.9 of the Collett (2003) for more
#'details. The  event time in this example is right censored.
#'
#'@format A data frame with 36 observations on the following 4 variables:
#'  \describe{ \item{\code{nephrectomy}}{indicator on whether or not the patient
#'  had recived a nephrectomy} \item{\code{age}}{age group at the time of
#'  diagnosis. 1 = <60, 2 = 60-70, 3 = >70.} \item{\code{time}}{observed time.}
#'  \item{\code{status}}{status of the observed time. 0 = censored, 1 = event.}
#'  }
#'
#'@details The data uses time and status to represent the observed survival
#'  time. To fit into the icweib function, left and right endpoints of censoring
#'  interval need to be derived (see examples).
#'
#'@references Collett, D. (2003). \emph{Modelling Survival Data in Medical
#'  Research, Second Edition}, Texts in statistical science. Taylor & Francis.
#'
#'@examples
#'data(hyper)
#'## Derive left and right endpoints from time and status
#'hyper$left <- hyper$time
#'hyper$right <- ifelse(hyper$status==1, hyper$time, Inf)
#'
"hyper"

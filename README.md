Closely Related Mixture Distributions
================================

This package fits CREMID, an algorithm for comparison across 
closely related mixture distributions. 

### Install
The package can be installed on Linux and Mac using `devtools`:

```S
install.packages('devtools')
library('devtools')
devtools::install_github('cremid', 'jacsor')
```

### Use
There are three functions in this package, and their descriptions are provided
in the help files

```S
ans <- Fit(Y, C)
PlotDiff(ans)
ans.cal <- Calibrate(ans)
```
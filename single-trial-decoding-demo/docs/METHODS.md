# Mathematical implementation contract

Axes: X[N,P,T], external Y[N,R] in 1..K; B[R,K,P,T]. Internal labels are converted to 0..K−1 exactly once at pipeline ingress. All readout windows are absolute half-open times. Training includes all valid trials.

## Encoding and temporal generalization

C = Helmert(K).T, with CᵀC=I and Cᵀ1=0. D contains intercept, all R position contrast blocks, and original trial order. For target r, w_n=N/(K count_r[y_nr]); beta_r=(DᵀWD + .001 diag(0,1,…,1))⁻¹DᵀWZ. B_r=C beta_r,target. Before solving, target columns residualized on nuisance columns must have numerical rank K−1. A ridge solution is not evidence of identifiability.

Inner scalers are fitted only on unique inner-training trials × native baseline frames, ddof=0; near-zero SD becomes 1. Standardization precedes two-frame binning. Conditional residuals remove the fitted intercept, other target blocks and trial order, not the current target block. Inner-validation labels are allowed only here inside outer-training data.

Each training target/time fits an inverse-class-frequency weighted neuron mean/variance and standardized centroids. Scores are centroid·standardized(x)−½||centroid||². Variances ≤10⁻¹² use the median of other valid variances, or 1. All R² directions are tested over all validation times; class hit counts are accumulated before averaging class accuracy.

## Data-driven priors

H_rs=max((TG_rs−1/K)/(1−1/K),0), gate_r(t)=sigmoid(clip((t−onset_r)/.05,−60,60)). For every unordered pair, cross support symmetrizes H_rs and H_srᵀ, multiplies a Gaussian kernel on onset-relative temporal difference (SD .10s) and both onset gates. Row/column sums accumulate into the two ranks, divided by R−1.

Private support is the nonnegative residual of symmetrized within-target row sums after a nonnegative scalar regression on shared support, then onset-gated. Both supports are Gaussian-smoothed (SD .75 bins, nearest boundary), clipped nonnegative and row L2 normalized; bin weights repeat twice then renormalize. Zero rows raise an explicit failure.

## Alternating joint fit

B_r ≈ a_r E + b_r R_r, with all a,b≥0. Fixed-gain templates solve the common (RT)×(R+1) temporal design with ridge 10⁻⁶; each template is position-centered. Private template vectors shrink along vec(E) by .25/1.25 = .2, not by projection onto E's full row span.

For profiles j=[a_1…a_R,b_1…b_R], energy weights are mean(E²), repeated R times, followed by mean(R_r²), floored at float64 epsilon. The objective is:

- L_data = Σ_r ||a_r E+b_r R_r−B_r||² / (RKPT)
- L_prior = λ_prior Σ_jt energy_j(profile_j−prior_j)² / (2RT)
- L_smooth = λ_smooth Σ_jt energy_j(Δ²profile_j)² / (2R(T−2))
- L_recur = λ_recur energy_E mean_(r<s) Σ_t w_rs(t)[a_r(t)−a_s(t+shift_rs)]²

w is sqrt(pE_r pE_s) on valid onset-aligned pairs, thresholded above 10⁻¹² and normalized to sum one. All unordered pairs participate. Onset differences must be integer samples; no implicit interpolation. Defaults λ=(1,2,1). L-BFGS-B uses exact analytic gradients, nonnegative bounds, maxiter=300, ftol=gtol=10⁻⁹, maxls=40.

After profile optimization, shared profiles divide by sqrt(Σ_r||a_r||²/R), with E multiplied by that scalar. Each private profile divides by its own norm while its template multiplies it. This preserves reconstruction before template refit. Templates are refit after normalization. Absolute objective change ≤10⁻⁷ max(1,|old|) after at least two iterations ends the algorithm; limit 80. Every optimizer status and final alternating status is recorded separately. Shrinkage and energy scaling mean this is not guaranteed descent of one global convex objective.

## Coordinates and fixed readout

Centered template SVD keeps s > max(s)10⁻¹⁰, no more than K−1 dimensions. Concatenate Q=[Q_E,Q_R1,…], and A=solve(QᵀQ+.25I,Qᵀ)ᵀ. These are simultaneous ridge coordinates, not independent Q projections.

Training window means are standardized with the outer scaler then multiplied by A. Entry pools R target windows with weights Nrows/(R K count_r[y]); private readouts use N/(K count_r[y]). Each readout uses weighted StandardScaler and L2 multinomial logistic regression with intercept, C=1, lbfgs, maxiter=2000, tol=10⁻⁶. A zero-dimensional block returns uniform probabilities and is marked.

Once latent scaling is folded into V,bias, every native test frame is decoded with frozen weights. No target label, conditional residual, learned time gain or test-trial mean enters prediction. Input-coordinate folded weights are Winput=(V A_jᵀ)/scale and binput=bias−Winput center. Coordinate and folded logits are tested to 10⁻⁸.

## Meaning and limitations

Space decomposition is operational and regularization dependent, not unique information attribution. The shared Entry readout is one location distribution at a time, not a simultaneous R-location posterior. Five folds share training data and are not five independent replications. Balanced accuracy and true-location probability are distinct. No significance stars or calibration/Bayesian claims are produced.

R=2 and R=3 use this same code; R=2 tests verify direct objective algebra/derivatives and end-to-end execution, not a byte-for-byte comparison against an unavailable historical implementation. R=3 adds all three recurrence/cross-TG pairs and the generalized temporal design, with no pairwise-model averaging.

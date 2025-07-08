# at-gpflow
Implementation of the adaptive transfer kernel of Cao et al. (2010) in GPFlow.

- [x] Implement AT kernel in GPFlow.
- [x] Implement AT objective in GPFlow.
- [x] Create likelihood capable of handling separate source/target variances.
- [x] Adapt GPR predict_f to handle separate source/target variances.
- [ ] Adapt TransferLikelihood - it now has some functions from SwitchedLikelihood that are not really adapted to the source/target situation yet.
- [ ] Tests.

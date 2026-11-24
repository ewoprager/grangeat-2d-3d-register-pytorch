Data graph updaters that make images showing how much each pixel contributes to a similarity metric.

| Updater                    | Writes                                                                            |
|----------------------------|-----------------------------------------------------------------------------------|
| `update_ncc_sim_met_image` | `ncc_sim_met_image`: product of the standardised moving and fixed images          |
| `update_gc_sim_met_image`  | `gc_sim_met_image`: the same for Sobel gradients, averaged over the two directions |

The mean of each image approximates the value of the metric, so bright regions are where the images agree. Both read
`moving_image` and `fixed_image`. The Sobel filter has no padding, so the gradient correlation image is two pixels smaller
in each direction.

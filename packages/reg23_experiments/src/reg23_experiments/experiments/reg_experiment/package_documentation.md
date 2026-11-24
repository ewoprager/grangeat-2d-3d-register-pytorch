The registration accuracy experiment: repeat a registration from random starting positions around a gold standard, and
record how the distance from the gold standard falls as the optimisation proceeds.

## Contents

| Module                       | Purpose                                                                                     |
|------------------------------|---------------------------------------------------------------------------------------------|
| `_reg_experiment`            | `ExperimentParametrisation` and the `reg_experiment` function                               |
| `_init_dadg`                 | Loads the CT and fills the data graph with initial values and updaters                      |
| `_setup`                     | Loads the saved transformations and X-ray configurations                                    |
| `_of_together`               | Objective functions that do not go through the graph, including the fused CUDA operator     |
| `_dadg_updaters.drr_reg`     | Loading, filtering, truncation and cropping: the part shared by every approach              |
| `_dadg_updaters.batched`     | Moving images and similarities for a batch of parameter vectors; used by experiments        |
| `_dadg_updaters.unbatched`   | The same for a single transformation, plus projected fiducials; used by the app             |
| `_dadg_updaters.grangeat`    | Sinograms, and unbatched Grangeat moving images and similarity                              |

## How a run works

1. The script builds an `ExperimentSetConfig` and calls `init_dadg`. This loads the CT, and adds updaters for either X-ray or
   synthetic target images.
2. `reg_experiment` is called for each configuration. It sets the graph nodes that the configuration controls, using equality
   checks so that unchanged inputs do not trigger reloading or recomputation.
3. Cropping and weighting are either updated on every objective function evaluation, by adding an updater, or every N
   iterations, as periodic behaviours of the registration.
4. For each repetition, a starting transformation is drawn at `starting_distance` from the gold standard and passed to
   `reg23_experiments.experiments.registration.register`.
5. The mean and standard deviation of the distance over the repetitions are returned for each iteration.

The objective function sets the `parameters` node to a batch of parameter vectors and reads `of_values`, or
`of_values_grangeat` if `grangeat` is set. Updating the cropping on every evaluation requires a batch size of 1.

## Options

| Option                         | Effect                                                                                   |
|--------------------------------|------------------------------------------------------------------------------------------|
| `downsample_level`             | Level of the CT pyramid; the X-ray is scaled to match                                    |
| `truncation_percent`           | How much of the CT to remove, split evenly between the top and bottom                    |
| `cropping_method`              | None, the projected bounds of the CT, or only rays through the full depth of the CT      |
| `weighting_method`, `weight_alpha` | How the scaling image is converted into similarity weights                           |
| `sim_metric`                   | Similarity metric, by name                                                               |
| `filter_method` and thresholds | Frequency filtering, applied equally to the CT and X-ray                                 |
| `iterations_per_..._update`    | Update interval for cropping and weighting; 0 means every evaluation                     |

## Data graph

The graphs below show the updaters added by `init_dadg` for X-ray targets. Rectangles name updaters, and the rest are nodes.
Nodes with no incoming arrow are set directly. `further_cropping` and `weight_images` are written either by an updater or
by a periodic behaviour.

### Shared by both approaches

```mermaid
flowchart TD
    set_xray_target_image[set_xray_target_image]
    xray_path(xray_path) ---> set_xray_target_image
    set_xray_target_image ---> source_distance(source_distance)
    set_xray_target_image ---> image_2d_full(image_2d_full)
    set_xray_target_image ---> image_2d_full_spacing(image_2d_full_spacing)
    set_xray_target_image ---> xray_sop_instance_uid(xray_sop_instance_uid)
%%
    load_ground_truth[load_ground_truth]
    saved_transformations(saved_transformations) ---> load_ground_truth
    ct_series_uid(ct_series_uid) ---> load_ground_truth
    xray_sop_instance_uid ---> load_ground_truth
    load_ground_truth ---> transformation_gt(transformation_gt)
%%
    load_base_cropping[load_base_cropping]
    saved_xray_reg_configs(saved_xray_reg_configs) ---> load_base_cropping
    xray_sop_instance_uid ---> load_base_cropping
    load_base_cropping ---> base_cropping(base_cropping)
    load_base_cropping ---> target_flipped(target_flipped)
%%
    combine_croppings[combine_croppings]
    base_cropping ---> combine_croppings
    further_cropping(further_cropping) ---> combine_croppings
    combine_croppings ---> cropping(cropping)
%%
    apply_filter_ct[apply_filter_ct]
    untruncated_ct_volume(untruncated_ct_volume) ---> apply_filter_ct
    ct_spacing(ct_spacing) ---> apply_filter_ct
    filter_method(filter_method) ---> apply_filter_ct
    lowpass_threshold(lowpass_threshold) ---> apply_filter_ct
    highpass_threshold(highpass_threshold) ---> apply_filter_ct
    apply_filter_ct ---> filtered_ct_volume(filtered_ct_volume)
%%
    apply_truncation[apply_truncation]
    filtered_ct_volume ---> apply_truncation
    truncation_percent(truncation_percent) ---> apply_truncation
    apply_truncation ---> ct_volumes(ct_volumes)
%%
    refresh_image_2d_scale_factor[refresh_image_2d_scale_factor]
    image_2d_full_spacing ---> refresh_image_2d_scale_factor
    downsample_level(downsample_level) ---> refresh_image_2d_scale_factor
    ct_spacing ---> refresh_image_2d_scale_factor
    refresh_image_2d_scale_factor ---> image_2d_scale_factor(image_2d_scale_factor)
    refresh_image_2d_scale_factor ---> fixed_image_spacing(fixed_image_spacing)
%%
    apply_filter_2d[apply_filter_2d]
    image_2d_full ---> apply_filter_2d
    image_2d_full_spacing ---> apply_filter_2d
    filter_method ---> apply_filter_2d
    lowpass_threshold ---> apply_filter_2d
    highpass_threshold ---> apply_filter_2d
    apply_filter_2d ---> filtered_image_2d(filtered_image_2d)
%%
    refresh_hyperparameter_dependent[refresh_hyperparameter_dependent]
    filtered_image_2d ---> refresh_hyperparameter_dependent
    image_2d_full_spacing ---> refresh_hyperparameter_dependent
    cropping ---> refresh_hyperparameter_dependent
    target_flipped ---> refresh_hyperparameter_dependent
    source_offset(source_offset) ---> refresh_hyperparameter_dependent
    image_2d_scale_factor ---> refresh_hyperparameter_dependent
    refresh_hyperparameter_dependent ---> cropped_target(cropped_target)
    refresh_hyperparameter_dependent ---> fixed_image_offset(fixed_image_offset)
    refresh_hyperparameter_dependent ---> translation_offset(translation_offset)
    refresh_hyperparameter_dependent ---> fixed_image_size(fixed_image_size)
```

### DRR-based registration

```mermaid
flowchart TD
    refresh_scaling_images[refresh_scaling_images]
    parameters(parameters) ---> refresh_scaling_images
    ct_volumes(ct_volumes) ---> refresh_scaling_images
    ct_spacing(ct_spacing) ---> refresh_scaling_images
    translation_offset(translation_offset) ---> refresh_scaling_images
    source_distance(source_distance) ---> refresh_scaling_images
    fixed_image_spacing(fixed_image_spacing) ---> refresh_scaling_images
    fixed_image_size(fixed_image_size) ---> refresh_scaling_images
    fixed_image_offset(fixed_image_offset) ---> refresh_scaling_images
    cropped_target(cropped_target) ---> refresh_scaling_images
    refresh_scaling_images ---> scaling_images(scaling_images)
    refresh_scaling_images ---> fixed_image(fixed_image)
%%
    project_moving_images[project_moving_images]
    parameters ---> project_moving_images
    ct_volumes ---> project_moving_images
    ct_spacing ---> project_moving_images
    source_distance ---> project_moving_images
    fixed_image_size ---> project_moving_images
    fixed_image_spacing ---> project_moving_images
    downsample_level(downsample_level) ---> project_moving_images
    translation_offset ---> project_moving_images
    fixed_image_offset ---> project_moving_images
    project_moving_images ---> moving_images(moving_images)
%%
    apply_sim_metric[apply_sim_metric]
    sim_metric(sim_metric) ---> apply_sim_metric
    moving_images ---> apply_sim_metric
    fixed_image ---> apply_sim_metric
    weight_images(weight_images) ---> apply_sim_metric
    apply_sim_metric ---> of_values(of_values)
```

### Grangeat-based registration

```mermaid
flowchart TD
    refresh_vif[refresh_vif]
    ct_volumes(ct_volumes) ---> refresh_vif
    ct_spacing(ct_spacing) ---> refresh_vif
    ct_series_uid(ct_series_uid) ---> refresh_vif
    sinogram_type(sinogram_type) ---> refresh_vif
    fixed_sinogram_size(fixed_sinogram_size) ---> refresh_vif
    downsample_level(downsample_level) ---> refresh_vif
    truncation_percent(truncation_percent) ---> refresh_vif
    filter_method(filter_method) ---> refresh_vif
    lowpass_threshold(lowpass_threshold) ---> refresh_vif
    highpass_threshold(highpass_threshold) ---> refresh_vif
    cache_directory(cache_directory) ---> refresh_vif
    refresh_vif ---> sinogram_size(sinogram_size)
    refresh_vif ---> vif(vif)
%%
    refresh_sinogram2d_grid[refresh_sinogram2d_grid]
    fixed_image_size(fixed_image_size) ---> refresh_sinogram2d_grid
    fixed_image_offset(fixed_image_offset) ---> refresh_sinogram2d_grid
    fixed_image_spacing(fixed_image_spacing) ---> refresh_sinogram2d_grid
    refresh_sinogram2d_grid ---> sinogram2d_grid_unshifted(sinogram2d_grid_unshifted)
    refresh_sinogram2d_grid ---> sinogram2d_grid(sinogram2d_grid)
%%
    refresh_sinogram2d[refresh_sinogram2d]
    fixed_image(fixed_image) ---> refresh_sinogram2d
    source_distance(source_distance) ---> refresh_sinogram2d
    fixed_image_spacing ---> refresh_sinogram2d
    sinogram2d_grid_unshifted ---> refresh_sinogram2d
    refresh_sinogram2d ---> sinogram2d(sinogram2d)
%%
    resample_for_moving_image_grangeat[resample_for_moving_image_grangeat]
    parameters(parameters) ---> resample_for_moving_image_grangeat
    vif ---> resample_for_moving_image_grangeat
    sinogram2d_grid ---> resample_for_moving_image_grangeat
    source_distance ---> resample_for_moving_image_grangeat
    translation_offset(translation_offset) ---> resample_for_moving_image_grangeat
    fixed_image_offset ---> resample_for_moving_image_grangeat
    resample_for_moving_image_grangeat ---> moving_images_grangeat(moving_images_grangeat)
%%
    apply_sim_metric_grangeat[apply_sim_metric_grangeat]
    sim_metric(sim_metric) ---> apply_sim_metric_grangeat
    moving_images_grangeat ---> apply_sim_metric_grangeat
    sinogram2d ---> apply_sim_metric_grangeat
    apply_sim_metric_grangeat ---> of_values_grangeat(of_values_grangeat)
```

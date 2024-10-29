import {Rect, LandmarkRect} from "../common/types.js";

class SSDParams {
    constructor(conf_threshold, iou_threshold, num_loc, num_class, num_prior_box, max_show_box_num) {
        this._conf_threshold = conf_threshold;
        this._iou_threshold = iou_threshold;
        this._num_class = num_class;
        this._num_loc = num_loc;
        this._num_prior_box = num_prior_box;
        this._max_show_box_num = max_show_box_num;
    }

    get num_class() {
        return this._num_class;
    }

    set num_class(cls) {
        this._num_class = cls;
    }

    get num_loc() {
        return this._num_loc;
    }

    set num_loc(locs) {
        this._num_loc = locs;
    }

    get conf_threshold() {
        return this._conf_threshold;
    }

    set conf_threshold(thres) {
        this._conf_threshold = thres;
    }

    get iou_threshold() {
        return this._iou_threshold;
    }

    set iou_threshold(thres) {
        this._iou_threshold = thres;
    }

    get num_prior_box() {
        return this._num_prior_box;
    }

    set num_prior_box(priors) {
        this._num_prior_box = priors;
    }

    get max_show_box_num() {
        return this._max_show_box_num;
    }
}

class PriorBoxParams {
    constructor(feature_map_sizes, image_shape, steps, min_sizes, max_sizes, ars, variance, clip) {
        this._feature_map_sizes = feature_map_sizes;
        this._image_shape = image_shape;
        this._steps = steps;
        this._min_sizes = min_sizes;
        this._max_sizes = max_sizes;
        this._aspect_ratios = ars;
        this._variance = variance;
        this._clip = clip;
    }

    get feature_map_sizes() {
        return this._feature_map_sizes;
    }

    // feature_map_size array
    set feature_map_sizes(sizes) {
        this._feature_map_sizes = sizes;
    }

    set image_shape(shape) {
        this._image_shape = shape;
    }

    get image_shape() {
        return this._image_shape;
    }

    set steps(steps) {
        this._steps = steps;
    }

    get steps() {
        return this._steps;
    }

    set min_sizes(sizes) {
        this._min_sizes = sizes;
    }

    get min_sizes() {
        return this._min_sizes;
    }

    set max_sizes(sizes) {
        this._max_sizes = sizes;
    }

    get max_sizes() {
        return this._max_sizes;
    }

    set aspect_ratios(ars) {
        this._aspect_ratios = ars;
    }

    get aspect_ratios() {
        return this._aspect_ratios;
    }

    get variance() {
        return this._variance;
    }

    set variance(va) {
        this._variance = va;
    }

    get clip() {
        return this._clip;
    }

    set clip(clip) {
        this._clip = clip;
    }

}

class PriorBox {
    constructor(l, t, w, h, box_id, score) {
        this._left = l;
        this._top = t;
        this._width = w;
        this._height = h;
        this._prior_box_id = box_id;
        this._conf_score = score;
        this._max_conf_score = score; // temp variable
    }

    get left() {
        return this._left;
    }

    set left(l) {
        this._left = l;
    }

    get top() {
        return this._top;
    }

    set top(t) {
        this._top = t;
    }

    get width() {
        return this._width;
    }

    set width(w) {
        this._width = w;
    }

    get height() {
        return this._height;
    }

    set height(h) {
        this._height = h;
    }

    get conf_score() {
        return this._conf_score;
    }

    set conf_score(score) {
        this._conf_score = score;
    }

    get max_conf_score() {
        return this._max_conf_score;
    }

    set max_conf_score(score) {
        this._max_conf_score = score;
    }

    get prior_box_id() {
        return this._prior_box_id;
    }

    set prior_box_id(id) {
        this._prior_box_id = id;
    }

}

class PriorBoxInfo {
    constructor(prior_box_params) {
        this._box_params = prior_box_params;
        this._prior_box_locs = new Array();
    }

    get box_params() {
        return this._box_params;
    }

    set box_params(params) {
        this._box_params = params;
    }

    checkParams() {
        console.assert(this.box_params.feature_map_sizes.length == this.box_params.min_sizes.length &&
            this.box_params.feature_map_sizes.length == this.box_params.max_sizes.length &&
            this.box_params.feature_map_sizes.length == this.box_params.steps.length &&
            this.box_params.feature_map_sizes.length == this.box_params.aspect_ratios.length);
    }

    checkVariance() {
        this.box_params.variance.forEach(function (variance) {
            console.assert(variance > 0);
        });
    }

    checkStep() {
        for (let i = 0; i < this.box_params.feature_map_sizes; i++) {
            console.assert(Math.abs((this.box_params.feature_map_sizes[i] * this.box_params.steps[i]) - this.box_params.image_shape) < 0.00001);
        }
    }

    //eg. (16x16 + 8x8 + 5x5 + 3x3 + 1x1)*(4+2) 4:aspect_rario size(eg. 1/2,1/3,2,3) 2:aspect_rario size(eg. s_k = min_size, s_k = sqrt(min_size*max_size)), when ar = 1
    getPriorBoxFeatureDataLength() {
        this.checkParams();
        this.checkStep();
        this.checkVariance();

        let feature_length = 0;
        for (let fmap_id = 0; fmap_id < this.box_params.feature_map_sizes.length; fmap_id++) {
            let f_k = this.box_params.feature_map_sizes[fmap_id];
            feature_length += (2 + this.box_params.aspect_ratios[fmap_id].length * 2) * f_k * f_k;
        }

        return feature_length;
    }

    getPriorBoxesInfo() {
        let feature_length = this.getPriorBoxFeatureDataLength();
        let BOX_LOC_NUM = 4;//cx,cy,w,h
        this._prior_box_locs = new Array(BOX_LOC_NUM);
        // this._prior_box_locs.forEach(function (elem,index,array) {
        //    array[index] = new Array(feature_length);
        // });

        for (let i = 0; i < this._prior_box_locs.length; i++) {
            this._prior_box_locs[i] = new Array(feature_length);
        }
        ;


        let prior_box_id = 0;
        for (let fmap_id = 0; fmap_id < this.box_params.feature_map_sizes.length; fmap_id++) {
            let feature_map_size = this.box_params.feature_map_sizes[fmap_id];
            for (let h = 0; h < feature_map_size; h++) {
                for (let w = 0; w < feature_map_size; w++) {
                    let f_k = this.box_params.image_shape / this.box_params.steps[fmap_id];
                    let cx = (w + 0.5) / f_k;
                    let cy = (h + 0.5) / f_k;
                    let s_k1 = this.box_params.min_sizes[fmap_id];
                    let s_k2 = this.box_params.max_sizes[fmap_id];
                    //aspect_ratio = 1
                    let prior_box_locs = new Array(BOX_LOC_NUM);
                    prior_box_locs[0] = cx;
                    prior_box_locs[1] = cy;
                    prior_box_locs[2] = s_k1;
                    prior_box_locs[3] = s_k1;

                    this._prior_box_locs.forEach(function (elem, index, array) {
                        array[index][prior_box_id] = prior_box_locs[index];
                    });
                    prior_box_id++;

                    //aspect_ratio = 1 of sqrt(s_k1*s_k2)
                    let s_k_prime = Math.sqrt(s_k1 * s_k2);

                    prior_box_locs[0] = cx;
                    prior_box_locs[1] = cy;
                    prior_box_locs[2] = s_k_prime;
                    prior_box_locs[3] = s_k_prime;

                    this._prior_box_locs.forEach(function (elem, index, array) {
                        array[index][prior_box_id] = prior_box_locs[index];
                    });
                    prior_box_id++;

                    //rest aspect_rario(eg. 1/2,1/3,2,3)
                    for (let ar_id = 0; ar_id < this.box_params.aspect_ratios[fmap_id].length; ar_id++) {
                        let ar_sqrt = Math.sqrt(this.box_params.aspect_ratios[fmap_id][ar_id]);

                        prior_box_locs[0] = cx;
                        prior_box_locs[1] = cy;
                        prior_box_locs[2] = s_k1 * ar_sqrt;
                        prior_box_locs[3] = s_k1 / ar_sqrt;

                        this._prior_box_locs.forEach(function (elem, index, array) {
                            array[index][prior_box_id] = prior_box_locs[index];
                        });

                        prior_box_id++;

                        prior_box_locs[0] = cx;
                        prior_box_locs[1] = cy;
                        prior_box_locs[2] = s_k1 / ar_sqrt;
                        prior_box_locs[3] = s_k1 * ar_sqrt;

                        this._prior_box_locs.forEach(function (elem, index, array) {
                            array[index][prior_box_id] = prior_box_locs[index];
                        });

                        prior_box_id++;
                    }
                }
            }

        }
        //clip to [0,1]
        if (this.box_params.clip) {
            this._prior_box_locs.forEach(function (loc_elem, loc_id, loc_array) {
                loc_array[loc_id].forEach(function (box_elem, box_id, box_array) {
                    box_array[box_id] = Math.max(Math.min(box_array[box_id], 1.0), 0.0);
                })
            });
        }
        return this._prior_box_locs;
    }

}


class SSD {
    constructor(ssd_params, prior_box_params) {
        this._ssd_params = ssd_params;
        this._prior_box_params = prior_box_params;
        this._prior_box_info = new PriorBoxInfo(this._prior_box_params);
        this._ssd_prior_boxes_info = this._prior_box_info.getPriorBoxesInfo();
        this._prior_box_num = 0;
    }

    get ssd_params() {
        return this._ssd_params;
    }

    get prior_box_params() {
        return this._prior_box_params;
    }

    get ssd_prior_boxes_info() {
        return this._ssd_prior_boxes_info;
    }

    get prior_box_num() {
        return this._prior_box_num;
    }

    calcAxisOverlap(min0, min1, max0, max1) {
        let l_val = Math.max(min0, min1);
        let r_val = Math.min(max0, max1);
        return (r_val - l_val);
    }

    calBoxOverlapArea(box0, box1) {
        let min_x0 = box0.left;
        let min_y0 = box0.top;
        let max_x0 = box0.left + box0.width;
        let max_y0 = box0.top + box0.height;

        let min_x1 = box1.left;
        let min_y1 = box1.top;
        let max_x1 = box1.left + box1.width;
        let max_y1 = box1.top + box1.height;

        let x_overlap = this.calcAxisOverlap(min_x0, min_x1, max_x0, max_x1);
        let y_overlap = this.calcAxisOverlap(min_y0, min_y1, max_y0, max_y1);

        if (x_overlap < 0 || y_overlap < 0)
            return 0;

        return (x_overlap * y_overlap);

    }

    calcIOURatio(box0, box1) {

        let min_x0 = box0.left;
        let min_y0 = box0.top;
        let max_x0 = box0.left + box0.width;
        let max_y0 = box0.top + box0.height;

        let min_x1 = box1.left;
        let min_y1 = box1.top;
        let max_x1 = box1.left + box1.width;
        let max_y1 = box1.top + box1.height;

        let overlap_area = this.calBoxOverlapArea(box0, box1);
        let box0_area = box0.width * box0.height;
        let box1_area = box1.width * box1.height;
        let union_area = box0_area + box1_area - overlap_area;
        //return (overlap_area/union_area);

        let iou0 = overlap_area / box0_area;
        let iou1 = overlap_area / box1_area;

        return Math.max(iou0, Math.max(iou1, (overlap_area / union_area)));
    }

    nonMaximumSuppression(prior_boxes, iou_threshold) {
        //discard any remaining box with IoU >= threshold
        for (let box_id = 0; box_id < this._prior_box_num; box_id++) {
            if (prior_boxes[box_id].max_conf_score < 0.00001)
                continue;

            for (let next_box_id = box_id + 1; next_box_id < this._prior_box_num; next_box_id++) {
                let iou_ratio = this.calcIOURatio(prior_boxes[box_id], prior_boxes[next_box_id]);
                if (iou_ratio > iou_threshold) {
                    if (prior_boxes[box_id].max_conf_score < prior_boxes[next_box_id].max_conf_score) {
                        //discard the prior box with the box_id
                        prior_boxes[box_id].max_conf_score = 0;
                        break;
                    }
                    //or discard the prior box with the next_box_id
                    prior_boxes[next_box_id].max_conf_score = 0;
                }
            }
        }
    }

    detection(loc_feature, conf_feature) {

        this._prior_box_num = 0;
        let prior_box_id = -1;

        let ssd_prior_boxes = new Array();
        let feat_conf_id_start = 0;
        // get confidence
        for (let fmap_id = 0; fmap_id < this.prior_box_params.feature_map_sizes.length; fmap_id++) {

            let width = this.prior_box_params.feature_map_sizes[fmap_id];
            let height = this.prior_box_params.feature_map_sizes[fmap_id];
            let channel = this.ssd_params.num_class * this.ssd_params.num_prior_box;

            for (let h = 0; h < height; h++) {
                for (let w = 0; w < width; w++) {
                    for (let c = 0; c < channel; c += this.ssd_params.num_class) {
                        let sum_exp = 0.0;
                        let cls_conf = new Float32Array(this.ssd_params.num_class);
                        for (let cls_id = 0; cls_id < this.ssd_params.num_class; cls_id++) {
                            let g_index = feat_conf_id_start + (h * width + w) * channel + c + cls_id;
                            cls_conf[cls_id] = Math.exp(conf_feature[g_index]);
                            sum_exp += cls_conf[cls_id];
                        }

                        for (let i = 0; i < cls_conf.length; i++) {
                            cls_conf[i] /= sum_exp;
                        }

                        prior_box_id++;
                        // if num_class == 2, 0:background 1:human face
                        cls_conf = cls_conf.slice(1);
                        //pick the box with the largest probability output as a prediction and the probability is larger than threshold
                        let max_loc_id = cls_conf.indexOf(Math.max(...cls_conf));
                        // console.log("confidence = ",cls_conf[max_loc_id]);
                        if (cls_conf[max_loc_id] > this.ssd_params.conf_threshold) {
                            let prior_box = new PriorBox(0, 0, 0, 0, prior_box_id, cls_conf[max_loc_id]);
                            ssd_prior_boxes.push(prior_box);
                            this._prior_box_num++;
                        }
                    }
                }
            }
            feat_conf_id_start += width * height * channel;
        }

        console.assert(feat_conf_id_start == conf_feature.length, "feature map conf data length mismatch");

        if (this._prior_box_num == 0)
            return null;

        let cur_prior_box_id = 0;
        let ssd_prior_boxes_id = 0;
        let feat_loc_id_start = 0;
        // get box pos
        for (let fmap_id = 0; fmap_id < this.prior_box_params.feature_map_sizes.length; fmap_id++) {

            let width = this.prior_box_params.feature_map_sizes[fmap_id];
            let height = this.prior_box_params.feature_map_sizes[fmap_id];
            let channel = this.ssd_params.num_loc * this.ssd_params.num_prior_box;


            for (let h = 0; h < height; h++) {
                for (let w = 0; w < width; w++) {
                    for (let c = 0; c < channel; c += this.ssd_params.num_loc) {
                        try {
                            if ((ssd_prior_boxes_id < this._prior_box_num) && (cur_prior_box_id == ssd_prior_boxes[ssd_prior_boxes_id].prior_box_id)) {
                                let g_index = feat_loc_id_start + (h * width + w) * channel + c;
                                ssd_prior_boxes[ssd_prior_boxes_id].left = loc_feature[g_index];
                                ssd_prior_boxes[ssd_prior_boxes_id].top = loc_feature[g_index + 1];
                                ssd_prior_boxes[ssd_prior_boxes_id].width = loc_feature[g_index + 2];
                                ssd_prior_boxes[ssd_prior_boxes_id].height = loc_feature[g_index + 3];

                                ssd_prior_boxes_id++;
                                // if (ssd_prior_boxes_id >= this._prior_box_num) break;
                            }
                        } catch (e) {
                            console.log(e);
                        }
                        cur_prior_box_id++;
                    }
                }
            }
            feat_loc_id_start += width * height * channel;
        }

        console.assert(feat_loc_id_start == loc_feature.length, "feature map conf data length mismatch");

        //get prior box info
        let prior_box_locs = this.ssd_prior_boxes_info;
        let cx_array = prior_box_locs[0];
        let cy_array = prior_box_locs[1];
        let width_array = prior_box_locs[2];
        let height_array = prior_box_locs[3];

        for (let box_id = 0; box_id < this._prior_box_num; box_id++) {
            //variance
            let tx = 0.1 * ssd_prior_boxes[box_id].left;
            let ty = 0.1 * ssd_prior_boxes[box_id].top;
            let tw = 0.2 * ssd_prior_boxes[box_id].width;
            let th = 0.2 * ssd_prior_boxes[box_id].height;

            let g_prior_box_id = ssd_prior_boxes[box_id].prior_box_id;

            let w = Math.exp(tw) * width_array[g_prior_box_id];
            let h = Math.exp(th) * height_array[g_prior_box_id];

            let cx = tx * width_array[g_prior_box_id] + cx_array[g_prior_box_id];
            let cy = ty * height_array[g_prior_box_id] + cy_array[g_prior_box_id];

            ssd_prior_boxes[box_id].left = cx - 0.5 * w;
            ssd_prior_boxes[box_id].top = cy - 0.5 * h;
            ssd_prior_boxes[box_id].width = w;
            ssd_prior_boxes[box_id].height = h;

        }

        //non maximum suppression algorithm
        this.nonMaximumSuppression(ssd_prior_boxes, this.ssd_params.conf_threshold);

        ssd_prior_boxes.sort(function (first, second) {
            return (second.max_conf_score - first.max_conf_score);
        });

        let prior_box_output = new Array();
        let show_box_num = 0;
        let IMG_SIZE = this._prior_box_params._image_shape;
        for (let box_id = 0; box_id < this._prior_box_num; box_id++) {
            let prior_box = ssd_prior_boxes[box_id];
            if (ssd_prior_boxes[box_id].max_conf_score > 0.1) {

                let landmark_rect = new LandmarkRect(prior_box.left * IMG_SIZE, prior_box.top * IMG_SIZE, prior_box.width * IMG_SIZE, prior_box.height * IMG_SIZE, prior_box.conf_score);
                prior_box_output.push(landmark_rect);
                show_box_num++;
                if (show_box_num >= this.ssd_params.max_show_box_num) break;
            }

        }

        return prior_box_output;
    }

}

export {SSD, SSDParams, PriorBox, PriorBoxParams, PriorBoxInfo}



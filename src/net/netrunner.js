import {makeTexture2D, updateTexture2D, log_output} from "../common/common.js";
import {Tensor} from "../common/tensor.js";
import {Transform} from "../op/transform.js";
import {Vector2D,Keypoint} from "../common/types.js";
import {findCandidateKeypoints} from "../common/utils.js";
import {SSD,SSDParams,PriorBox,PriorBoxParams} from "./ssd.js";

class NetRunner {
    constructor(gl, packed, input_scale = 1.0, input_trans = 0.0) {
        this._gl = gl;
        this._packed = packed;
        this._width = 0;
        this._height = 0;
        this._texture = null;
        this._input_tensor = this._makeInputTensor();
        this._trans = new Transform(this._gl, packed, input_scale, input_trans);
    }

    set_input(channel, height, width, modelInChannel, modelInHeight, modelInWidth, net) {
        this._width = width;
        this._height = height;

        // let dim_in = Array.of(channel, height, width);
        let dim_in = Array.of(modelInChannel, modelInHeight, modelInWidth);
        this._input_tensor = new Tensor(this._gl, this._packed, dim_in, null, null,2, false);

        net.setInputTensor(this._input_tensor);
        net.allocMemory();
    }

    run(data, net) {

        if(!this._texture) {
            this._texture = makeTexture2D(this._gl, this._width, this._height, this._gl.UNSIGNED_BYTE, data, false);
        } else {
            updateTexture2D(this._gl, this._texture, this._width, this._height, this._gl.UNSIGNED_BYTE, data);
        }
        // unsigned to float
        this._trans.run(this._texture, this._input_tensor,0);
        // now run the network
        // let run_st = performance.now();
        net.run();
        // let run_ed = performance.now();
        // console.log("model run time："+(run_ed - run_st) + "ms")
    }

    _makeInputTensor() {
        let dims_in = Array.of(3, 256, 256);
        return new Tensor(this._gl, this._packed, dims_in, null);
    }
}

export {NetRunner}
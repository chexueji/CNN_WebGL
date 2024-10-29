import {Tensor} from "../common/tensor.js";
import {DEBUG, MODELVERIFY} from "../common/common.js";

class Node {
    constructor() {
        this._input_node = new Array();
        this._input_tensor = new Array();
        this._output_tensor = null;
        this._operator = null;
        this._layerIndex = null;
    }

    run() {
        if (MODELVERIFY)
        //if(false)
        {
            if (this._layerIndex == 1) {
                let tex_width = this._input_tensor[0].storage.textureWidth;
                let tex_height = this._input_tensor[0].storage.textureHeight;
                let block_horizon = this._input_tensor[0].storage.blockHorizon;
                let tex_data = new Float32Array(tex_width * tex_height);
                let channel = this._input_tensor[0].dim(0);
                let height = this._input_tensor[0].dim(1);
                let width = this._input_tensor[0].dim(2);
                for (let c = 0; c < channel; c++) {
                    let block_w = parseInt(c % block_horizon);
                    let block_h = parseInt(c / block_horizon);
                    let block_w_start = block_w * width;
                    let block_h_start = block_h * height;

                    for (let h = 0; h < height; h++) {
                        let row = block_h_start + h;
                        for (let w = 0; w < width; w++) {
                             tex_data[row * tex_width +  block_w_start + w] = window.layer_feature_data[0][c * height * width + h * width + w];
                            //tex_data[row * tex_width + block_w_start + w] = window.layer_feature_data[0][c * height * width + h * width + w];
                        }
                    }
                }

                let input_tensor = new Tensor(this._input_tensor[0]._gl, false, this._input_tensor[0].dims(), tex_data);
                let input_tensors = new Array();
                input_tensors.push(input_tensor);
                this._operator.run(input_tensors, this._output_tensor, this._layerIndex);

            } else if (this._layerIndex == 3000000) {
                let tex_width = this._input_tensor[0].storage.textureWidth;
                let tex_height = this._input_tensor[0].storage.textureHeight;
                let block_horizon = this._input_tensor[0].storage.blockHorizon;
                let tex_data = new Float32Array(tex_width * tex_height);
                let channel = this._input_tensor[0].dim(0);
                let height = this._input_tensor[0].dim(1);
                let width = this._input_tensor[0].dim(2);
                for (let c = 0; c < channel; c++) {
                    let block_w = parseInt(c % block_horizon);
                    let block_h = parseInt(c / block_horizon);
                    let block_w_start = block_w * width;
                    let block_h_start = block_h * height;

                    for (let h = 0; h < height; h++) {
                        let row = block_h_start + h;
                        for (let w = 0; w < width; w++) {
                            // tex_data[row * tex_width +  block_w_start + w] = window.layerData[c * height * width + h * width + w];
                            tex_data[row * tex_width + block_w_start + w] = window.layer_feature_data[3][c * height * width + h * width + w];
                        }
                    }
                }

                let input_tensor = new Tensor(this._input_tensor[0]._gl, false, this._input_tensor[0].dims(), tex_data);
                let input_tensors = new Array();
                input_tensors.push(input_tensor);
                this._operator.run(input_tensors, this._output_tensor, this._layerIndex);
                //let rawData = this._output_tensor.rawData();
                //console.save(rawData,"test_output.json");
            } else {
                this._operator.run(this._input_tensor, this._output_tensor, this._layerIndex);
            }

            if(true) {
                let raw_data = this._output_tensor.rawData();
                let feature_data = window.layer_feature_data[this._layerIndex];
                console.assert(raw_data.length == feature_data.length, 'feature/run data length mismatch');
                let max_error = 0;
                let avg_error = 0;
                let sum_error = 0;
                let max_error_index = 0;
                let error_num = 0;

                feature_data.forEach(function (value, index) {
                    let error = Math.abs(value - raw_data[index]);
                    if (error >= 0.000001) {
                        error_num++;
                    }
                    sum_error += error;
                    if (error > max_error) {
                        max_error = error;
                        max_error_index = index;
                    }
                });

                avg_error = sum_error / feature_data.length;

                let test_result = 'test' + ((max_error >= 0.001) ? ' fail' : ' pass');

                console.log(test_result, ', layer id:', this._layerIndex, ', op:', this._operator.constructor.name, ', avg:', avg_error.toFixed(10), ', max:', max_error.toFixed(10), 'at (', max_error_index, '), errors:', error_num, "(error >= 0.000001)");
            }

        } else {
            this._operator.run(this._input_tensor, this._output_tensor, this._layerIndex);
        }

    }

    allocMemory() {
        for (let idx = 0; idx < this._input_node.length; idx++) {
            this._input_tensor.push(this._input_node[idx].outputTensor);
        }

        this._output_tensor = this._operator.allocMemory(this._input_tensor);
    }

    directRun() {
        this._input_tensor.clear();
        for (let idx = 0; idx < this._inputs.length; idx++) {
            this._input_tensor.push(this._input_node.output());
        }

        this._output_tensor = this._operator.directRun(this._input_tensor);
    }

    addInput(input) {
        this._input_node.push(input);
    }

    getInputs() {
        return this._input_node;
    }

    set operator(oper) {
        this._operator = oper;
    }

    get operator() {
        return this._operator;
    }

    set LayerId(id) {
        this._layerIndex = id;
    }

    get LayerId() {
        return this._layerIndex;
    }

    output() {
        return this._output_tensor;
    }

    set outputTensor(output) {
        this._output_tensor = output;
    }

    get outputTensor() {
        return this._output_tensor;
    }

    readPixel() {

    }
}

class Network {
    constructor() {
        //this._data_node = null;
        this._nodes = new Array(); // 0 is data node.
        this._fused_nodes = new Array();
        this._op_names = null;
    }

    run() {
        for (let idx = 1; idx < this._nodes.length; idx++) {
            this._nodes[idx].run();
        }
    }

    rebuild() {
        let oriNodes = this._nodes;
        for (let idx = 0; idx < oriNodes.length;) {
            //this._nodes = this._nodes.reduce(function (nodes, curNode, idx, oriNodes) {


            if ((oriNodes[idx]._operator instanceof Conv2D) &&
                (oriNodes[idx + 1] && oriNodes[idx + 1]._operator instanceof BatchNorm) &&
                (oriNodes[idx + 2] && oriNodes[idx + 2]._operator instanceof Relu) && oriNodes[idx + 3]) {

                let nodes = oriNodes[idx + 3]._input_node.reduce(function (result, curNode, i) {
                    if ((curNode._operator instanceof Relu) &&
                        (oriNodes[curNode.LayerId - 2]._operator instanceof Conv2D) &&
                        (oriNodes[curNode.LayerId - 1]._operator instanceof BatchNorm)) {
                        result.push(oriNodes[curNode.LayerId - 2]);
                        return result;
                    }
                    result.push(curNode);
                    return result;
                }, []);
                oriNodes[idx + 3]._input_node = nodes;
                this._fused_nodes.push(oriNodes[idx]);
                idx += 3;
            } else {
                let nodes = oriNodes[idx]._input_node.reduce(function (result, curNode, i) {
                    if ((curNode._operator instanceof Relu) &&
                        (oriNodes[curNode.LayerId - 2]._operator instanceof Conv2D) &&
                        (oriNodes[curNode.LayerId - 1]._operator instanceof BatchNorm)) {
                        result.push(oriNodes[curNode.LayerId - 2]);
                        return result;
                    }
                    result.push(curNode);
                    return result;
                }, []);
                oriNodes[idx]._input_node = nodes;
                this._fused_nodes.push(oriNodes[idx]);
                idx++;
            }
        }

        this._nodes = this._fused_nodes;
    }

    directRun() {
        for (let idx = 1; idx < this._nodes.length; idx++) {
            this._nodes[idx].directRun();
        }
    }

    allocMemory() {
        for (let idx = 1; idx < this._nodes.length; idx++) {
            //console.log(idx);
            this._nodes[idx].allocMemory();
        }
    }

    allocMemory_fused() {
        for (let idx = 1; idx < this._fused_nodes.length; idx++) {
            try {
                this._fused_nodes[idx].allocMemory();
            } catch (e) {
                console.log(e);
            }
        }
    }

    setInputData(gl, packed, data, channel, height, width) {
        if (this._nodes.length) {
            this._nodes[0].outputTensor = new Tensor(gl, packed, [channel, height, width], data, null);
            //let dat = this._nodes[0].outputTensor.rawData();
            //console.log(dat);
        }
    }

    setInputTensor(input_tensor) {
        if (this._nodes.length) {
            this._nodes[0].outputTensor = input_tensor;
        }
    }

    getOutput(index) {
        return this._nodes[index].outputTensor;
    }

    getOutputNodes() {
        let nodes = new Array();
        let nodes_tag = new Map();
        for (let i = 0; i < this._nodes.length; i++) {
            let inputs = this._nodes[i].getInputs();
            for (let j = 0; j < inputs.length; j++) {
                nodes_tag.set(inputs[j], 1);
            }
        }
        for (let i = 0; i < this._nodes.length; i++) {
            if (nodes_tag.get(this._nodes[i]) == undefined) {
                nodes.push(this._nodes[i]);
            }
        }

        return nodes;
    }

    getOutputByName(name) {
        let index = -1; //this._op_names.get(name);
        for (let i = this._op_names.length - 1; i >= 0; i--) {
            if (this._op_names[i].localeCompare(name) == 0) {
                index = i;
                break;
            }
        }
        return this.getOutput(index);
    }

    get dataNode() {
        return this._nodes[0];
    }

    set dataNode(nd) {
        this._nodes[0] = nd;
    }

    get opNames() {
        return this._op_names;
    }

    set opNames(value) {
        this._op_names = value;
    }

    addNode(nd) {
        this._nodes.push(nd);
    }

    node(index) {
        return this._nodes[index];
    }

    clear() {
        this._nodes = new Array();
        ;
    }

}

export {Node, Network}

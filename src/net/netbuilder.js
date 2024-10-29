import {DEBUG} from "../common/common.js";
import {Conv2D, DepthwiseConv2D} from "../op/convolution.js";
import {Relu, HardTanh, Sigmoid} from "../op/nonlinear.js";
import {MaxPool, AvePool} from "../op/pool.js";
import {Tensor} from "../common/tensor.js";
import {Node, Network} from "./network.js";
import {BatchNorm} from "../op/batchnorm.js";
import {Upsampling_Nearest, Upsampling_Bilinear} from "../op/upsampling.js";
import {ElementWise_Add, ElementWise_CAdd, ElementWise_CMul, ElementWise_Avg, ElementWise_SliceMul} from "../op/elementwise.js";
import {View, Permute} from "../op/reshape.js";
import {Concat2D_2, Concat2D_3, Concat1D} from "../op/concat.js";
import {Reorg} from "../op/reorg.js";

class Attribute {
    constructor() {
        this._key = 0;
        this._value = null;
        this._type = 0;
    }

    get key() {
        return this._key;
    }

    set key(v) {
        this._key = v;
    }

    get value() {
        return this._value;
    }

    set value(v) {
        this._value = v;
    }

    get type() {
        return this._type;
    }

    set type(v) {
        this._type = v;
    }
}

class Layer {
    constructor() {
        this._attributes = new Map();
    }

    addAttribute(attrib) {
        this._attributes.set(attrib.key, attrib);
    }

    get attributes() {
        return this._attributes;
    }

    attributeByKey(key) {
        return this._attributes.get(key);
    }

    type() {
        let attrib = this.attributeByKey(0);
        if (attrib) {
            return attrib.value;
        }
        return -1;
    }
}

class Downloader {
    constructor() {
        this._op_names = new Array();
        this._layers = new Map();
    }

    get opNames() {
        return this._op_names;
    }

    get layers() {
        return this._layers;
    }

    down(path, netbuilder) {
        let self = this;
        let xhr = new XMLHttpRequest();
        xhr.responseType = 'arraybuffer';
        xhr.onload = function () {
            self.parseModel(this.response, netbuilder);
        }
        xhr.open("GET", path, true);
        xhr.send();
    }

    parseModel(buffer, netbuilder) {
        let offset = 0;
        let version = new Int8Array(buffer, offset, 4);
        if (DEBUG) {
            console.log(version[0]);
            console.log(version[1]);
            console.log(version[2]);
            console.log(version[3]);
        }

        offset += 4;
        offset = this._loadHeader(buffer, offset);

        let index = 0;
        while (offset < buffer.byteLength) {
            let op_header = new Int32Array(buffer, offset, 3);
            //let key = op_header[0];
            let type = op_header[1];
            let length = op_header[2];
            //console.log(op_header);
            switch (type) {
                case 0:
                    this._layers.set(index, this._decodeNested(buffer, offset, length));
                    break;
                case 1:
                    break;
                default:
                    break;
            }

            offset += length;
            index++;
        }

        netbuilder.build(this);
    }

    _decodeNested(buffer, offset, length) {
        let layer = new Layer();
        let pos = 12;
        offset += 12;
        while (pos < length) {
            let head_arr = new Int32Array(buffer, offset, 3);
            let attr_type = head_arr[1];
            let attr_length = head_arr[2];
            let attrib = new Attribute();
            attrib.key = head_arr[0];
            attrib.type = attr_type;
            switch (attr_type) {
                case 0: // nested
                    break;
                case 1: // int32
                    attrib.value = this._decodeInt32(buffer, offset);
                    break;
                case 2: // uint32
                    attrib.value = this._decodeUint32(buffer, offset);
                    break;
                case 3: // int8
                    attrib.value = this._decodeInt8(buffer, offset);
                    break;
                case 4: // uint8
                    attrib.value = this._decodeUint8(buffer, offset);
                    break;
                case 5: // float32
                    attrib.value = this._decodeFloat32(buffer, offset);
                    break;
                case 6: // string
                    break;
                case 7: // file
                    break;
                case 8: // array
                    attrib.value = this._decodeArray(buffer, offset, attr_length);
                    break;
                default:
                    break;
            }

            layer.addAttribute(attrib);
            pos += attr_length;
            offset += attr_length;
        }

        return layer;
    }

    _decodeInt32(buffer, offset) {
        let val_arr = new Int32Array(buffer, offset + 12, 1);
        //console.log(val);

        return val_arr[0];
    }

    _decodeUint32(buffer, offset) {
        let val_arr = new Uint32Array(buffer, offset + 12, 1);
        //console.log(val);

        return val_arr[0];
    }

    _decodeInt8(buffer, offset) {
        let val_arr = new Int8Array(buffer, offset + 12, 1);
        //console.log(val);
        return val_arr[0];
    }

    _decodeUint8(buffer, offset) {
        let val_arr = new Uint8Array(buffer, offset + 12, 1);
        //console.log(val);
        return val_arr[0];
    }

    _decodeFloat32(buffer, offset) {
        let val_arr = new Uint8Array(buffer, offset + 12, 1);
        //console.log(val);
        return val_arr[0];
    }

    _decodeArray(buffer, offset) {
        let sub_arr = new Int32Array(buffer, offset + 12, 1);
        let len_arr = new Uint32Array(buffer, offset + 16, 1);
        let data_offset = offset + 20;
        let data_arr = null;
        switch (sub_arr[0]) {
            case 1: // int32
            {
                data_arr = new Int32Array(buffer, data_offset, len_arr[0]);
                //console.log(data_arr);
                break;
            }
            case 2: // uint32
            {
                data_arr = new Uint32Array(buffer, data_offset, len_arr[0]);
                //console.log(data_arr);
                break;
            }
            case 3: // int8
            {
                data_arr = new Int8Array(buffer, data_offset, len_arr[0]);
                //console.log(data_arr);
                break;
            }
            case 4: // uint8
            {
                data_arr = new Uint8Array(buffer, data_offset, len_arr[0]);
                //console.log(data_arr);
                break;
            }
            case 5: // float32
            {
                data_arr = new Float32Array(buffer, data_offset, len_arr[0]);
                //console.log(data_arr);
                break;
            }
                break;
            default:
                console.error("Error: unrecognized array type.");
        }

        return data_arr;
    }

    _loadHeader(buffer, offset) {
        let num_arr = new Uint32Array(buffer, offset, 1);
        offset += 4;

        for (let i = 0; i < num_arr[0]; i++) {
            let index_arr = new Int32Array(buffer, offset, 1);
            offset += 4;
            let len_arr = new Uint32Array(buffer, offset, 1);
            offset += 4;
            let sz = parseInt(Math.ceil(len_arr[0] / 4.0)) * 4;
            let name_arr = new Uint8Array(buffer, offset, sz);
            offset += sz;

            let name = String.fromCharCode.apply(String, name_arr);
            this._op_names.push(name);
            //console.log(name);
        }
        return offset;
    }
}

function toUnicode(theString) {
    let unicodeString = '';

    for (let i = 0; i < theString.length; i++) {
        let theUnicode = theString.charCodeAt(i).toString(16).toUpperCase();

        while (theUnicode.length < 4) {
            theUnicode = '0' + theUnicode;
        }

        theUnicode = '\\u' + theUnicode;
        unicodeString += theUnicode;
    }

    return unicodeString;
}

class NetworkBuilder {
    constructor(gl, packed) {
        this._gl = gl;
        this._packed = packed;
        this._net = null;
        this._callback_func = null;
    }

    get network() {
        return this._net;
    }

    buildFromUrl(path, callback_func) {
        this._callback_func = callback_func;
        let downloader = new Downloader();
        downloader.down(path, this);
    }

    build(loader) {
        this._net = new Network();
        this._net.opNames = loader.opNames;

        let layers = loader.layers;
        for (let [index, layer_param] of layers) {
            let layer_type = layer_param.type();
            let node = null;
            switch (layer_type) {
                case 0: // data
                {
                    node = this._createDataNode(layer_param);
                    if (DEBUG) console.log("Data " + index);
                    break;
                }
                case 1: // convolution
                {
                    node = this._createConvNode(layer_param);
                    if (DEBUG) console.log("Conv2D " + index);
                    break;
                }
                case 2: // pool
                {
                    node = this._createPoolNode(layer_param);
                    if (DEBUG) console.log("Pooling " + index);
                    break;
                }
                case 3: // bn
                {
                    node = this._createBatchNormNode(layer_param);
                    if (DEBUG) console.log("BatchNorm " + index);
                    break;
                }
                case 4: // non-linear
                {
                    node = this._createNonLinearNode(layer_param);
                    if (DEBUG) console.log("Relu " + index);
                    break;
                }
                case 5: // de-conv
                {
                    console.log("Deconv is NOT supported.")
                    break;
                }
                case 6: // up-sampling
                {
                    if (index === 161)
                        console.log();
                    node = this._createUpsampleNode(layer_param);
                    if (DEBUG) console.log("Upsampling " + index);
                    break;
                }
                case 7: // concat
                {
                    node = this._createConcatNode(layer_param);
                    if (DEBUG) console.log("Concat " + index);
                    break;
                }
                case 8: // elementwise
                {
                    node = this._createElementwiseNode(layer_param);
                    if (DEBUG) console.log("Elementwise " + index);
                    break;
                }
                case 9: // depthwise-conv
                {
                    node = this._createDwConvNode(layer_param);
                    if (DEBUG) console.log("Depthwise Conv " + index);
                    break;
                }
                case 10: // up-pooling
                    console.log("Un-pooling is NOT supported.")
                    break;
                case 11: // linear
                    console.log("Linear is NOT supported.")
                    break;
                case 12: // id
                    console.log("Identity is NOT supported.")
                    break;

                case 13: // reshape
                {
                    node = this._createReshapeNode(layer_param);
                    // if (DEBUG) console.log("Reshape " + index);
                    break;
                }
                case 18: // reorg
                {
                    node = this._createReorgNode(layer_param);
                    if (DEBUG) console.log("Reorg " + index);
                    break;
                }
                default:
                    console.log("Not supported oprators." + layer_type);
            }
            node.LayerId = index;
            this._net.addNode(node);
        }

        if (this._callback_func) {
            this._callback_func();
        }
    }

    _createDataNode(layer_param) {
        //this._net.dataNode = new Node();
        let data_node = new Node();

        return data_node;
    }

    _createConvNode(layer_param) {
        let conv_node = new Node();
        let conv_op = new Conv2D(this._gl, this._packed);
        //let conv_op = new Conv2DQuant(this._gl, this._packed);
        conv_node.operator = conv_op;
        for (let [type, attrib] of layer_param.attributes) {
            switch (type) {
                case 0: // op type
                    break;
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        //console.log(in_node_id);
                        conv_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }

                case 13: // output channel
                    conv_op.channel_out = attrib.value;
                    break;
                case 11: // kernel size
                    conv_op.kernel = attrib.value;
                    break;
                case 8: // stride
                    conv_op.stride = attrib.value;
                    break;
                case 9: // padding
                    conv_op.padding = attrib.value;
                    break;
                case 10: // dilation
                    conv_op.dilation = attrib.value;
                    break;
                case 12: // input channel
                    break;
                case 19: // weight
                    //conv_op.setWeight()
                    break;
                case 20: // bias // todo
                    //conv_op.setBias();
                    break;
                case 56: // weight max min
                    //conv_op.weight_min = attrib.value[1];
                    //conv_op.weight_max = attrib.value[0];
                    break;
                case 58: // conv groups
                    conv_op.groups = attrib.value;
                    //console.log("conv groups:" + attrib.key);
                    break;
                default:
                    console.log("Warninig: un-solved convolution attribute:" + attrib.key);
            }
        }

        let weight_attrib = layer_param.attributeByKey(19);
        if (weight_attrib) {
            conv_op.setWeight(weight_attrib.value);
        }

        let bias_attrib = layer_param.attributeByKey(20);
        if (bias_attrib) {
            conv_op.setBias(bias_attrib.value);
        }

        return conv_node;
    }

    _createPoolNode(layer_param) {
        let pool_node = new Node();
        let pool_op = null;
        let pool_type_attrib = layer_param.attributeByKey(1);
        if (pool_type_attrib) {
            switch (pool_type_attrib.value) {

                case 0: // max
                    pool_op = new MaxPool(this._gl, this._packed);
                    break;
                case 1: // avc
                    pool_op = new AvePool(this._gl, this._packed);
                    break;
                case 2: // sto
                    console.error("Error: st pool not supported.")
                    break;
                default:
                    console.error("Error: un-recognized pooling type.")
                    break;
            }
        }
        pool_node.operator = pool_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 0: // op type
                    break;
                case 1: // pool type
                    break;
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        pool_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                case 12: // input channel
                    break;
                case 13: // output channel
                    pool_op.channel_out = attrib.value;
                    break;
                case 11: // kernel size
                    pool_op.kernel = attrib.value;
                    break;
                case 8: // stride
                    pool_op.stride = attrib.value;
                    break;
                case 9: // padding
                    pool_op.padding = attrib.value;
                    break;
                case 10: // dilation
                    pool_op.dilation = attrib.value;
                    break;
                default:
                    console.log("Warining: un-solved pooling attribute:" + attrib.key);
            }
        }

        return pool_node;
    }

    _createBatchNormNode(layer_param) {
        let bn_node = new Node();
        let bn_oper = new BatchNorm(this._gl, this._packed);
        bn_node.operator = bn_oper;

        let mean = null, variance = null, weight = null, bias = null;
        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 0: // op type
                    break;
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        bn_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                case 13: // output channel
                    break;
                case 19: // weight
                {
                    weight = attrib.value;
                    break;
                }
                case 20: // bias
                {
                    bias = attrib.value;
                    break;
                }
                case 30: // mean
                {
                    mean = attrib.value;
                    break;
                }
                case 31: // variance
                {
                    variance = attrib.value;
                    break;
                }
                default:
                    console.warn("Warning: un-recognized attribute:" + key);
            }
        }
        let channel_in = mean.length;
        let bn_param_data = new Float32Array(channel_in * 4);
        if (this._packed) {
            for (let c = 0; c < channel_in; c++) {
                let index = c * 4;
                bn_param_data[index] = mean[c];
                bn_param_data[index + 1] = variance[c];
                bn_param_data[index + 2] = weight[c];
                bn_param_data[index + 3] = bias[c];
            }
        } else {
            bn_param_data.set(mean);
            bn_param_data.set(variance, channel_in);
            bn_param_data.set(weight, channel_in * 2);
            if (bias) {
                bn_param_data.set(bias, channel_in * 3);
            }
        }

        bn_oper.setParameter(channel_in, bn_param_data);

        return bn_node;
    }

    _createNonLinearNode(layer_param) {
        let nl_node = new Node();
        let nl_op = null;
        let nl_type_attrib = layer_param.attributeByKey(2);
        if (nl_type_attrib) {
            switch (nl_type_attrib.value) {
                case 0: // relu
                    nl_op = new Relu(this._gl, this._packed);
                    break;
                case 1: // sigmoid
                    nl_op = new Sigmoid(this._gl, this._packed);
                    break;
                case 2: // Tanh
                    break;
                case 3: // softmax
                case 4:
                    console.error("Error: Not implemented non-linear type:" + nl_type_attrib.value);
                    break;
                case 5: // hardtanh
                    let nl_value_attrib = layer_param.attributeByKey(56);
                    let max = nl_value_attrib.value[0];
                    let min = nl_value_attrib.value[1];
                    nl_op = new HardTanh(this._gl, this._packed, max, min);
                    break;
                case 6: // prelu
                    break;
                default:
                    console.warn("Warning: un-solved non-linear type:" + nl_type_attrib.value);
            }
        }
        nl_node.operator = nl_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 0: // op type
                    break
                case 2: // non-linear type
                    break;
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        nl_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                case 12: // input channel
                    break;
                case 13: // output channel
                    break;
                default:
                    console.warn("Warining: un-solved non-linear attribute type:" + key);
                    break;
            }
        }

        return nl_node;
    }

    _createUpsampleNode(layer_param) {
        let us_node = new Node();

        let us_op = null;
        let us_type_attrib = layer_param.attributeByKey(3);
        if (us_type_attrib) {
            if (us_type_attrib.value == 0) {
                us_op = new Upsampling_Nearest(this._gl, this._packed);
            } else {
                us_op = new Upsampling_Bilinear(this._gl, this._packed);
            }
        }
        us_node.operator = us_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 0:
                    break;
                case 3:
                    break; // upsampling type
                case 6: // input
                    for (let in_node_id of attrib.value) {
                        us_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                case 48:
                case 45:
                    us_op.upsampleScale = attrib.value;
                    break;
                case 60: // align_corners
                    us_op.alignCorners = attrib.value;
                    break;
                default:
                    break;
            }
        }

        return us_node;
    }

    _createConcatNode(layer_param) {
        let concat_node = new Node();
        let input_num = layer_param.attributeByKey(6).value.length;
        let concat_op;
        let concat_dim;
        switch (input_num) {
            case 2:
                concat_dim = layer_param.attributeByKey(59).value;
                concat_op = new Concat2D_2(this._gl, this._packed, concat_dim);
                break;
            case 3:
                concat_dim = layer_param.attributeByKey(59).value;
                concat_op = new Concat2D_3(this._gl, this._packed, concat_dim);
                break;
            case 5:
                concat_op = new Concat1D(this._gl, this._packed);
                break;
            default:
                concat_op = new Concat2D_2(this._gl, this._packed);
        }
        concat_node.operator = concat_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (attrib.key) {
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        concat_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return concat_node;
    }

    _createElementwiseNode(layer_param) {
        let elt_node = new Node();
        let elt_op = null;
        let elt_type_attrib = layer_param.attributeByKey(4);
        if (elt_type_attrib) {
            switch (elt_type_attrib.value) {
                case 0: // add
                    elt_op = new ElementWise_Add(this._gl, this._packed);
                    if (DEBUG) console.log("ElementWise_Add");
                    break;
                case 1: // mul
                    console.error("Error: un-solved element-wise type:" + elt_type_attrib.value);
                    break;
                case 2: // cadd
                {
                    elt_op = new ElementWise_CAdd(this._gl, this._packed);
                    let elt_value_attrib = layer_param.attributeByKey(28);
                    elt_op.inputConst = elt_value_attrib.value;
                    if (DEBUG) console.log("ElementWise_CAdd");
                    break;
                }
                case 3: // cmul
                    elt_op = new ElementWise_CMul(this._gl, this._packed);
                    elt_op.inputConst = layer_param.attributeByKey(28).value;
                    if (DEBUG) console.log("ElementWise_CMul");
                    break;
                case 8: // avg
                    elt_op = new ElementWise_Avg(this._gl, this._packed);
                    if (DEBUG) console.log("ElementWise_Avg");
                    break;
                case 9: // slice mul
                    elt_op = new ElementWise_SliceMul(this._gl, this._packed);
                    if (DEBUG) console.log("ElementWise_SliceMul");
                    break;
                default:
                    console.error("Error: un-solved element-wise type:" + elt_type_attrib.value);
                    break;
            }
        }
        elt_node.operator = elt_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        elt_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return elt_node;
    }

    _createDwConvNode(layer_param) {
        let conv_node = new Node();
        let conv_op = new DepthwiseConv2D(this._gl, this._packed);
        conv_node.operator = conv_op;
        for (let [type, attrib] of layer_param.attributes) {
            switch (type) {
                case 0: // op type
                    break;
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        conv_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }

                case 13: // output channel
                    conv_op.channel_out = attrib.value;
                    break;
                case 11: // kernel size
                    conv_op.kernel = attrib.value;
                    break;
                case 8: // stride
                    conv_op.stride = attrib.value;
                    break;
                case 9: // padding
                    conv_op.padding = attrib.value;
                    break;
                case 10: // dilation
                    conv_op.dilation = attrib.value;
                    break;
                case 12: // input channel
                    break;
                case 19: // weight
                    //conv_op.setWeight()
                    break;
                case 20: // bias // todo
                    //conv_op.setBias();
                    break;
                case 56: // weight max/min
                    break;
                case 58: // conv groups
                    conv_op.groups = attrib.value;
                    // console.log("conv groups:" + attrib.key);
                    break;
                default:
                    console.log("Warninig: un-solved convolution attribute:" + attrib.key);
            }
        }

        let weight_attrib = layer_param.attributeByKey(19);
        if (weight_attrib) {
            conv_op.setWeight(weight_attrib.value);
        }

        let bias_attrib = layer_param.attributeByKey(20);
        if (bias_attrib) {
            conv_op.setBias(bias_attrib.value);
        }

        return conv_node;
    }

    _createReshapeNode(layer_param) {
        let reshape_node = new Node();
        let reshape_op = null;
        let reshape_type_attrib = layer_param.attributeByKey(54);
        if (reshape_type_attrib) {
            switch (reshape_type_attrib.value) {
                case 0: // view
                    let reshape_value_attrib = layer_param.attributeByKey(55);
                    let dims = reshape_value_attrib.value;
                    if (dims.length == 2) {
                        dims = Array.of(1, dims[0], dims[1]);
                    }
                    reshape_op = new View(this._gl, this._packed, dims);
                    if (DEBUG) console.log("View");
                    break;
                case 1: // permute
                {
                    let reshape_value_attrib = layer_param.attributeByKey(55);
                    let dim_trans_idx = reshape_value_attrib.value;
                    if (dim_trans_idx.length == 4 && dim_trans_idx[0] == 0) {
                        dim_trans_idx = Array.of(dim_trans_idx[1] - 1, dim_trans_idx[2] - 1, dim_trans_idx[3] - 1);
                    } else if (dim_trans_idx == 3) {

                    } else {
                        console.error("Permute Op dims trans size invalidate");
                    }
                    reshape_op = new Permute(this._gl, this._packed, dim_trans_idx);
                    if (DEBUG) console.log("Permute");
                    break;
                }
                default:
                    console.error("Error: un-solved reshape type:" + reshape_type_attrib.value);
                    break;
            }
        }

        reshape_node.operator = reshape_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        reshape_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return reshape_node;
    }

    _createReorgNode(layer_param) {
        let reorg_node = new Node();
        let reorg_op = new Reorg(this._gl, this._packed);

        reorg_node.operator = reorg_op;

        for (let [key, attrib] of layer_param.attributes) {
            switch (key) {
                case 6: // input
                {
                    for (let in_node_id of attrib.value) {
                        reorg_node.addInput(this._net.node(in_node_id));
                    }
                    break;
                }
                default:
                    break;
            }
        }
        return reorg_node;
    }

    makeInputTexture(gl) {
        // HWC format
        let height = 256, width = 256, channel = 3;
        let image_data = new Float32Array(channel * height * width);
        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                let pixel_index = h * width + w;
                let index = h * width * channel + w * channel;
                image_data[index] = pixel_index;
                image_data[index + 1] = pixel_index;
                image_data[index + 2] = pixel_index;
            }
        }

        // create texture
        let texture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, width, height, 0, gl.RGB, gl.FLOAT, image_data);
        gl.bindTexture(gl.TEXTURE_2D, null);

        return texture;
    }
}

export {NetworkBuilder}
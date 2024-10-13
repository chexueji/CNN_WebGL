import {getMin2exp, getHorizonBlock, makeFbo} from "./common.js";

class StorageBase {
    constructor(gl, dims, data, texture, linear) {
        this._gl = gl;
        this._texture_height = 0;
        this._texture_width = 0;
        this._block_horizon = 0;
        this._block_vertical = 0;
        this._texture = null;
        this._fbo = null;
        if (texture) {
            this._texture = texture;
        } else {
            this._texture = this._makeTexture(dims, data, linear);
            this._fbo = makeFbo(gl, this._texture);
        }
    }

    get texture() {
        return this._texture;
    }

    get textureWidth() {
        return this._texture_width;
    }

    get textureHeight() {
        return this._texture_height;
    }

    get blockHorizon() {
        return this._block_horizon;
    }

    get blockVertical() {
        return this._block_vertical;
    }

    bind() {
        this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, this._fbo);
    }

    unbind() {
        this._gl.bindFramebuffer(this._gl.FRAMEBUFFER, null);
    }
}

class Storage extends StorageBase {
    constructor(gl, dims, data, texture = null, texture_type = 2, linear = false) {
        super(gl, dims, data, texture, linear);
    }

    copy(gl, other) {
        //gl.copyTexImage2D()
    }

    setData(data) {
        this._gl.bindTexture(this._gl.TEXTURE_2D, this._texture);
        this._gl.texImage2D(this._gl.TEXTURE_2D, 0, this._gl.R32F,
            this._texture_width, this._texture_height,
            0, this._gl.RED, this._gl.FLOAT, data);
        this._gl.bindTexture(this._gl.TEXTURE_2D, null);
    }

    data() {
        this.bind();
        let output_data = new Float32Array(4 * this._texture_width * this._texture_height);
        this._gl.readBuffer(gl.COLOR_ATTACHMENT0);
        this._gl.readPixels(0, 0, this._texture_width, this._texture_height, this._gl.RGBA, this._gl.FLOAT, output_data);
        this.unbind();

        let dat = new Float32Array(this._texture_width * this._texture_height);
        for (let i = 0; i < this._texture_width * this._texture_height; i++) {
            dat[i] = output_data[i * 4];
        }

        this.unbind();
        return dat;
    }

    rawData(channel, height, width) {
        this.bind();
        let output_data = new Float32Array(4 * this._texture_width * this._texture_height);
        this._gl.readBuffer(this._gl.COLOR_ATTACHMENT0);
        this._gl.readPixels(0, 0, this._texture_width, this._texture_height, this._gl.RGBA, this._gl.FLOAT, output_data);
        this.unbind();

        let fdata = new Float32Array(channel * height * width);
        for (let c = 0; c < channel; c++) {
            let block_w = parseInt(c % this._block_horizon);
            let block_h = parseInt(c / this._block_horizon);
            let block_w_start = block_w * width;
            let block_h_start = block_h * height;

            for (let h = 0; h < height; h++) {
                let row = block_h_start + h;
                for (let w = 0; w < width; w++) {
                    fdata[c * height * width + h * width + w] = output_data[(row * this._texture_width + block_w_start + w) * 4];
                }
            }
        }

        return fdata;
    }

    _makeTexture(dims, data, linear = false) {
        let ndata = null;
        if (dims.length == 3) {
            this._block_horizon = getMin2exp(dims[0]);
            this._block_vertical = dims[0] / this._block_horizon;
            this._texture_height = this._block_vertical * dims[1];
            this._texture_width = this._block_horizon * dims[2];
            ndata = data;
        } else if (dims.length == 4) {
            let channel_out = dims[0];
            let channel_in = dims[1];
            let kernel_h = dims[2];
            let kernel_w = dims[3];

            this._block_vertical = channel_out;
            this._block_horizon = channel_in;
            this._texture_height = this._block_vertical * kernel_h;
            this._texture_width = this._block_horizon * kernel_w;

            if (data) {
                ndata = new Float32Array(this._texture_height * this._texture_width);

                for (let h = 0; h < dims[0] * kernel_h; h++) {
                    let oc = parseInt(h / kernel_h);
                    let kh = h % kernel_h;
                    let input_offset = oc * channel_in * kernel_h * kernel_w;
                    for (let w = 0; w < channel_in * kernel_w; w++) {
                        let ic = parseInt(w / kernel_w);
                        let kw = w % kernel_w;
                        let index = input_offset + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                        ndata[h * channel_in * kernel_w + w] = data[index];
                    }
                }
            }
        }

        let texture = this._gl.createTexture();

        this._gl.bindTexture(this._gl.TEXTURE_2D, texture);
        this._gl.texParameteri(this._gl.TEXTURE_2D, this._gl.TEXTURE_MAG_FILTER, linear ? this._gl.LINEAR : this._gl.NEAREST);
        this._gl.texParameteri(this._gl.TEXTURE_2D, this._gl.TEXTURE_MIN_FILTER, linear ? this._gl.LINEAR : this._gl.NEAREST);
        this._gl.texImage2D(this._gl.TEXTURE_2D, 0, this._gl.R32F, this._texture_width,
            this._texture_height, 0, this._gl.RED, this._gl.FLOAT, ndata);
        this._gl.bindTexture(this._gl.TEXTURE_2D, null);

        return texture;
    }

}

class PackStroage extends StorageBase {
    constructor(gl, dims, data, texture = null, texture_type = 2) {
        super(gl, dims, data, texture);
    }

    setData(data) {
        this._gl.bindTexture(this._gl.TEXTURE_2D, this._texture);
        this._gl.texImage2D(this._gl.TEXTURE_2D, 0, this._gl.RGBA32F,
            this._texture_width, this._texture_height,
            0, this._gl.RGBA, this._gl.FLOAT, data);
        this._gl.bindTexture(this._gl.TEXTURE_2D, null);
    }

    rawData(channel, height, width) {
        this.bind();
        let output_data = new Float32Array(4 * this._texture_width * this._texture_height);
        this._gl.readBuffer(this._gl.COLOR_ATTACHMENT0);
        this._gl.readPixels(0, 0, this._texture_width, this._texture_height, this._gl.RGBA, this._gl.FLOAT, output_data);
        this.unbind();

        let fdata = new Float32Array(channel * height * width);
        for (let oc = 0; oc < channel; oc++) {
            let block_index = oc / 4;
            let c_in = oc % 4;
            let block_h = parseInt(block_index / this._block_horizon);
            let block_w = parseInt(block_index % this._block_horizon);
            let block_w_start = block_w * width * 4;
            let block_h_start = block_h * height;
            for (let oh = 0; oh < height; oh++) {
                let row = block_h_start + oh;
                for (let ow = 0; ow < width; ow++) {
                    fdata[oc * height * width + oh * width + ow] = output_data[row * this._texture_width * 4 + block_w_start + ow * 4 + c_in];
                }
            }
        }

        return fdata;
    }

    _makeTexture(dims, data, linear = false) {
        let ndata = null;
        let channel = dims[0];
        if (dims.length == 3) {
            if (channel == 3) {
                this._block_horizon = 1;
                this._block_vertical = 1;
                if (data) {
                    ndata = new Float32Array(dims[1] * dims[2] * 4); // h * w
                    for (let h = 0; h < dims[1]; h++) {
                        for (let w = 0; w < dims[2]; w++) {
                            let index = h * dims[2] + w;
                            ndata[index * 4] = data[index * 3];
                            ndata[index * 4 + 1] = data[index * 3 + 1];
                            ndata[index * 4 + 2] = data[index * 3 + 2];
                            ndata[index * 4 + 3] = 0;
                        }
                    }
                }
            } else {
                this._block_horizon = getHorizonBlock(channel);
                this._block_vertical = (channel / 4) / this._block_horizon;
                ndata = data;
            }

            this._texture_height = this._block_vertical * dims[1];
            this._texture_width = this._block_horizon * dims[2];
        } else if (dims.length == 4) {
            let channel_out = dims[0];
            let channel_in = dims[1];
            let kernel_h = dims[2];
            let kernel_w = dims[3];

            if (channel_in == 3) {
                this._block_vertical = channel_out;
                this._block_horizon = 1;
                this._texture_height = this._block_vertical * kernel_h;
                this._texture_width = this._block_horizon * kernel_w;

                if (data) {
                    ndata = new Float32Array(channel_out * 4 * kernel_h * kernel_w);
                    let kernel_size = kernel_w * kernel_h;
                    for (let oc = 0; oc < channel_out; oc++) {
                        let oc_offset0 = oc * 4 * kernel_h * kernel_w;
                        let oc_offset1 = oc * 3 * kernel_h * kernel_w;
                        for (let h = 0; h < kernel_h; h++) {
                            for (let w = 0; w < kernel_w; w++) {
                                let index = h * kernel_w + w;
                                ndata[oc_offset0 + index * 4] = data[oc_offset1 + index];
                                ndata[oc_offset0 + index * 4 + 1] = data[oc_offset1 + index + kernel_size];
                                ndata[oc_offset0 + index * 4 + 2] = data[oc_offset1 + index + kernel_size + kernel_size];
                                ndata[oc_offset0 + index * 4 + 3] = 0;
                            }
                        }
                    }
                }
            } else {
                this._block_vertical = channel_out;
                this._block_horizon = channel_in / 4;
                this._texture_height = this._block_vertical * kernel_h;
                this._texture_width = this._block_horizon * kernel_w;

                if (data) {
                    ndata = new Float32Array(this._texture_height * this._texture_width * 4);
                    for (let h = 0; h < this._texture_height; h++) {
                        let oc = parseInt(h / kernel_h);
                        let kh = h % kernel_h;

                        let input_offset = oc * channel_in * kernel_h * kernel_w;
                        for (let w = 0; w < this._texture_width; w++) {
                            let ic_index = parseInt(w / kernel_w);
                            let kw = w % kernel_w;
                            for (let c = 0; c < 4; c++) {
                                let ic = ic_index * 4 + c;
                                let index = input_offset + ic * kernel_h * kernel_w + kh * kernel_w + kw;
                                ndata[h * this._texture_width * 4 + w * 4 + c] = data[index];
                            }
                        }
                    }
                }
            }
        }

        let texture = this._gl.createTexture();

        this._gl.bindTexture(this._gl.TEXTURE_2D, texture);
        this._gl.texParameteri(this._gl.TEXTURE_2D, this._gl.TEXTURE_MAG_FILTER, linear ? this._gl.LINEAR : this._gl.NEAREST);
        this._gl.texParameteri(this._gl.TEXTURE_2D, this._gl.TEXTURE_MIN_FILTER, linear ? this._gl.LINEAR : this._gl.NEAREST);
        this._gl.texImage2D(this._gl.TEXTURE_2D, 0, this._gl.RGBA32F, this._texture_width,
            this._texture_height, 0, this._gl.RGBA, this._gl.FLOAT, ndata);
        this._gl.bindTexture(this._gl.TEXTURE_2D, null);

        return texture;
    }
}

class Tensor {
    constructor(context, packed, dims, data, texture = null, texture_type = 2, linear = false) {
        this._gl = context;
        this._dims = dims;
        this._strides = null;
        this._num = 0;
        this._computeStride();

        // alloc memory
        if (packed) {
            this._storage = new PackStroage(context, dims, data, texture, texture_type, linear);
        } else {
            this._storage = new Storage(context, dims, data, texture, texture_type, linear);
        }
    }

    dim(index) {
        return this._dims[index];
    }

    dims() {
        return this._dims;
    }

    stride(index) {
        return this._strides[index];
    }

    rawData() {

        return this._storage.rawData(this._dims[0], this._dims[1], this._dims[2]);

    }

    get storage() {
        return this._storage;
    }

    set storage(value) {
        this._storage = value;
    }

    flatten() {
        this._dims = Array.of(1, 1, this._num);
        this._strides = Array.of(1);
        this._storage = this.rawData();
    }

    bind() {
        this._storage.bind();
    }

    unbind() {
        this._storage.unbind();
    }

    _computeStride() {
        this._strides = new Array(this._dims.length);
        let stride = 1;
        for (let idx = this._dims.length - 1; idx >= 0; idx--) {
            this._strides[idx] = stride;
            stride *= this._dims[idx];
        }

        this._num = stride;
    }
}

export {Storage, Tensor}
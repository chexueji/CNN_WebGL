import {createShader, createProgram} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class Pool extends Operator {
    constructor(gl, packed, channel_out = 1, kernel = [2, 2], padding = [0, 0], stride = [2, 2], dilation = [1, 1]) {
        super(gl, packed);

        this._channel_out = channel_out;
        this._kernel = kernel;
        this._padding = padding;
        this._stride = stride;
        this._dilation = dilation;

        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._input_texture_loc = 0;
        //this._weight_texture_loc = 0;
        this._uniform_block_loc = 0;
        this._createProgram();
    }

    allocMemory(inputs) {
        let input = inputs[0];
        let input_storage = input.storage;
        let dims_in = input.dims();

        let nkh = this._kernel[0] + (this._kernel[0] - 1) * (this._dilation[0] - 1);
        let oh = parseInt((dims_in[1] + this._padding[0] * 2 - nkh) / this._stride[0]) + 1;

        let nkw = this._kernel[1] + (this._kernel[1] - 1) * (this._dilation[1] - 1);
        let ow = parseInt((dims_in[2] + this._padding[1] * 2 - nkw) / this._stride[1]) + 1;

        let dims_out = Array.of(this._channel_out, oh, ow);

        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;

        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let conv_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            dims_out[2], dims_out[1],//output width, height
            this._padding[0], this._padding[1],
            this._stride[0], this._stride[1], // pad, stride
            this._kernel[0], this._kernel[1],
            this._dilation[0], this._dilation[1], // kernel, dilation
            input_storage.blockHorizon, input_storage.blockVertical,
            dims_in[2], dims_in[1], //input width, height
            dims_in[0], -1, -1, -1]);
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, conv_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output, layerIndex) {
        let input = inputs[0];
        let in_storage = input.storage;
        let out_storage = output.storage;

        output.bind();
        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._unifrom_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("Pool(" + this._channel_in + ", " + this._channel_out +
                ",kernel_size=(" + this._kernel + "),stride=(" + this._stride + "),padding=(" + this._padding + "),stride=(" + this._stride + "),dilation=(" + this._dilation + ")");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    directRun(inputs) {
        let output_tensor = this.allocMemory(inputs);
        this.run(inputs, output_tensor);

        return output_tensor;
    }

    set kernel(kernel) {
        this._kernel = kernel;
    }

    get kernel() {
        return this._kernel;
    }

    set padding(padding) {
        this._padding = padding;
    }

    get padding() {
        return this._padding;
    }

    set stride(stride) {
        this._stride = stride;
    }

    get stride() {
        return this._stride;
    }

    set dilation(dilation) {
        this._dilation = dilation;
    }

    get dilation() {
        return this._dilation;
    }

    set channel_in(chn_in) {
        this._channel_in = chn_in;
    }

    get channel_in() {
        return this._channel_in;
    }

    set channel_out(chn_out) {
        this._channel_out = chn_out;
    }

    get channel_out() {
        return this._channel_out;
    }

    _createProgram() {
        let frag_source = null;
        if (this._packed) {
            frag_source = this._packFragShaderSource();
        } else {
            frag_source = this._fragShaderSource();
        }

        //let vert_shader = createShader(this._gl, this._gl.VERTEX_SHADER, vert_source);
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
    }

    _shaderCommon() {
        return "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "\n" +
            "uniform ConvParam {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "  ivec2 padding;\n" +
            "  ivec2 stride;\n" +
            "  ivec2 kernel;\n" +
            "  ivec2 dilation;\n" +
            "\n" +
            "  ivec2 in_image_num;\n" +
            "  ivec2 in_image_size;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n";
    }
}

class MaxPool extends Pool {
    constructor(gl, packed, channel_out = 1, kernel = [2, 2], padding = [0, 0], stride = [2, 2], dilation = [1, 1]) {
        super(gl, packed, channel_out, kernel, padding, stride, dilation);
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int kernel_size = kernel.x * kernel.y;\n" +
            "  ivec2 half_kernel = kernel / 2;\n" +
            "\n" +
            "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size)) * stride - padding;\n" +
            "\n" +
            "  ivec2 kernel_start = max(-image_coord_start, 0);\n" +
            "\n" +
            "  ivec2 kernel_end = min(in_image_size - image_coord_start, kernel);\n" +
            "\n" +
            "  ivec2 in_coord = ivec2(out_image_idx * in_image_size + image_coord_start); // h & w\n" +
            "\n" +
            "  float max_value = 0.0;\n" +
            "  for(int kh = kernel_start.y; kh < kernel_end.y; kh++)\n" +
            "  {\n" +
            "      for(int kw = kernel_start.x; kw < kernel_end.x; kw++)\n" +
            "      {\n" +
            "         max_value = max(texelFetch(input_texture, in_coord + ivec2(kw, kh), 0).x, max_value);\n" +
            "      }\n" +
            "  }\n" +
            "\n" +
            "  color = vec4(max_value, 0.0, 0.0, 0.0);\n" +
            "}";
    }

    _packFragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int kernel_size = kernel.x * kernel.y;\n" +
            "  ivec2 half_kernel = kernel / 2;\n" +
            "\n" +
            "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size)) * stride - padding;\n" +
            "\n" +
            "  ivec2 kernel_start = max(-image_coord_start, 0);\n" +
            "\n" +
            "  ivec2 kernel_end = min(in_image_size - image_coord_start, kernel);\n" +
            "\n" +
            "  ivec2 in_coord = ivec2(out_image_idx * in_image_size + image_coord_start); // h & w\n" +
            "\n" +
            "  color = vec4(0.0, 0.0, 0.0, 0.0);\n" +
            "  for(int kh = kernel_start.y; kh < kernel_end.y; kh++)\n" +
            "  {\n" +
            "      for(int kw = kernel_start.x; kw < kernel_end.x; kw++)\n" +
            "      {\n" +
            "         color = max(texelFetch(input_texture, in_coord + ivec2(kw, kh), 0), color);\n" +
            "      }\n" +
            "  }\n" +
            "}";
    }
}

class AvePool extends Pool {
    constructor(gl, channel_out = 1, kernel = [2, 2], padding = [0, 0], stride = [2, 2], dilation = [1, 1]) {
        super(gl, channel_out, kernel, padding, stride, dilation);
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int kernel_size = kernel.x * kernel.y;\n" +
            "  ivec2 half_kernel = kernel / 2;\n" +
            "\n" +
            "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size)) * stride - padding;\n" +
            "\n" +
            "  ivec2 kernel_start = max(-image_coord_start, 0);\n" +
            "\n" +
            "  ivec2 kernel_end = min(in_image_size - image_coord_start, kernel);\n" +
            "\n" +
            "  ivec2 in_coord = ivec2(out_image_idx * in_image_size + image_coord_start); // h & w\n" +
            "\n" +
            "  float sum = 0.0;\n" +
            "  for(int kh = kernel_start.y; kh < kernel_end.y; kh++)\n" +
            "  {\n" +
            "      for(int kw = kernel_start.x; kw < kernel_end.x; kw++)\n" +
            "      {\n" +
            "         sum += texelFetch(input_texture, in_coord + ivec2(kw, kh), 0).x;\n" +
            "      }\n" +
            "  }\n" +
            "\n" +
            "  sum /= float(kernel_size);\n" +
            "\n" +
            "  color = vec4(sum, 0.0, 0.0, 0.0);\n" +
            "}";
    }

    _packFragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int kernel_size = kernel.x * kernel.y;\n" +
            "  ivec2 half_kernel = kernel / 2;\n" +
            "\n" +
            "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size)) * stride - padding;\n" +
            "\n" +
            "  ivec2 kernel_start = max(-image_coord_start, 0);\n" +
            "\n" +
            "  ivec2 kernel_end = min(in_image_size - image_coord_start, kernel);\n" +
            "\n" +
            "  ivec2 in_coord = ivec2(out_image_idx * in_image_size + image_coord_start); // h & w\n" +
            "\n" +
            "  vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);\n" +
            "  for(int kh = kernel_start.y; kh < kernel_end.y; kh++)\n" +
            "  {\n" +
            "      for(int kw = kernel_start.x; kw < kernel_end.x; kw++)\n" +
            "      {\n" +
            "         sum += texelFetch(input_texture, in_coord + ivec2(kw, kh), 0);\n" +
            "      }\n" +
            "  }\n" +
            "\n" +
            "  sum /= float(kernel_size);\n" +
            "\n" +
            "  color = sum;\n" +
            "}";
    }
}

export {AvePool, MaxPool}
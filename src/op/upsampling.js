import {createShader, createProgram} from "../common/common.js";
import {Tensor} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class Upsampling extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._upsample_scale = 1;
        this._output_size = [];

        this._input_texture_loc = 0;
        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._uniform_block_loc = 0;
        this._align_corners = 0;

        this._createProgram();
    }

    allocMemory(inputs) {
        let input = inputs[0];
        let input_storage = input.storage;
        let dims_in = input.dims();//chw
        if (this._output_size.length == 0) {
            this._output_size = [dims_in[1] * this._upsample_scale,
                dims_in[2] * this._upsample_scale];
        }
        let dims_out = Array.of(dims_in[0], this._output_size[0], this._output_size[1]);// c,h,w

        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;
        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let conv_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            dims_out[2], dims_out[1],//output width, height
            input_storage.blockHorizon, input_storage.blockVertical,
            dims_in[2], dims_in[1], this._channel_in, -1, -1, -1]);//input width, height
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, conv_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output) {
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
            let result = output.rawData();
            console.log(result);
        }

        output.unbind();
    }

    directRun(inputs) {
        let output_tensor = this.allocMemory(inputs);
        this.run(inputs, output_tensor);

        return output_tensor;
    }

    get outputSize() {
        return this._output_size;
    }

    set outputSise(value) {
        this._output_size = value;
    }

    get upsampleScale() {
        return this._upsample_scale;
    }

    set upsampleScale(value) {
        this._upsample_scale = value;
    }

    get alignCorners() {
        return this._align_corners;
    }

    set alignCorners(value){
        this._align_corners = value;
    }

    _createProgram() {
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, this._fragShaderSource());
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'upsample_param');
    }

    _shaderCommon() {
        return "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform upsample_param {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "  ivec2 in_image_num;\n" +
            "  ivec2 in_image_size;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n";
    }
}

class Upsampling_Bilinear extends Upsampling {
    constructor(gl, packed) {
        super(gl, packed);
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size);\n" +
            "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image_size);\n" +
            "\n" +
            "  vec2 scale = vec2(in_image_size - 1) / vec2(out_image_size - 1);\n" +
            "  vec2 f_coord = vec2(image_coord) * scale;\n" +
            "  ivec2 i_coord_low = ivec2(f_coord);\n" +
            "\n" +
            "  vec2 factor_high = f_coord - vec2(i_coord_low);\n" +
            "  vec2 factor_low = 1.0 - factor_high;\n" +
            "\n" +
            "  ivec2 image_offset_in = ivec2(out_image_idx * in_image_size);\n" +
            "  color = (texelFetch(input_texture, image_offset_in + i_coord_low, 0) * factor_low.x +\n" +
            "          texelFetch(input_texture, image_offset_in + i_coord_low + ivec2(1, 0), 0) * factor_high.x) * factor_low.y +\n" +
            "          (texelFetch(input_texture, image_offset_in + i_coord_low + ivec2(0, 1), 0) * factor_low.x +\n" +
            "          texelFetch(input_texture, image_offset_in + i_coord_low + ivec2(1, 1), 0) * factor_high.x) * factor_high.y;\n" +
            "}";
    }
}

class Upsampling_Nearest extends Upsampling {
    constructor(gl, packed) {
        super(gl, packed);
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size);\n" +
            "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image_size);\n" +
            "\n" +
            "  vec2 scale = vec2(in_image_size - 1) / vec2(out_image_size - 1);\n" +
            "  vec2 f_coord = vec2(image_coord) * scale;\n" +
            "  ivec2 i_coord = ivec2(round(f_coord));\n" +
            "\n" +
            "  ivec2 image_offset_in = ivec2(out_image_idx * in_image_size);\n" +
            "\n" +
            "  color = texelFetch(input_texture, image_offset_in + i_coord, 0);\n" +
            "}";
    }
}

export {Upsampling_Nearest, Upsampling_Bilinear}
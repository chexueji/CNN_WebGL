import {createShader, createProgram, makeTypedTexture2D} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class BatchNorm extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._param_texture = null;
        this._input_texture_loc = 0;
        this._param_texture_loc = 0;
        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._uniform_block_loc = 0;
        this._createProgram();
    }

    allocMemory(inputs) {
        let input = inputs[0];
        //let input_storage = input.storage();
        let dims_in = input.dims();

        let output_tensor = new Tensor(this._gl, this._packed, dims_in);
        let output_storage = output_tensor.storage;
        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let conv_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
                                         dims_in[2], dims_in[1]]);//input width, height
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
        let vertexPosLocation = 0;
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
        // uniform
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._unifrom_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, this._param_texture);
        this._gl.uniform1i(this._param_texture_loc, 1);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if(DEBUG) {
            console.log("BatchNorm()");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex+"_output.json");
        }

        output.unbind();
    }

    directRun(inputs) {
        let output_tensor = this.allocMemory(inputs);
        this.run(inputs, output_tensor);

        return output_tensor;
    }

    setParameter(channel, data) {
        if(this._packed) {
            this._param_texture = makeTypedTexture2D(this._gl, channel, 1, this._gl.RGBA32F,
                                                     this._gl.RGBA, this._gl.FLOAT, data);
        } else {
            this._param_texture = makeTypedTexture2D(this._gl, channel, 4, this._gl.R32F,
                                                     this._gl.RED, this._gl.FLOAT, data);
        }
    }

    _shaderCommon() {
        return  "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform sampler2D param_texture;\n" +
            "uniform BnParam {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n";
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  vec4 data = texelFetch(input_texture, screen_coord, 0);\n" +
            "  vec4 mu = texelFetch(param_texture, ivec2(oc, 0), 0);\n" +
            "  vec4 epsilon = texelFetch(param_texture, ivec2(oc, 1), 0);\n" +
            "  vec4 alpha = texelFetch(param_texture, ivec2(oc, 2), 0);\n" +
            "  vec4 beta = texelFetch(param_texture, ivec2(oc, 3), 0);\n" +
            "\n" +
            "  vec4 result = (data - mu) / sqrt(epsilon+1E-5) * alpha + beta;\n" +
            "\n" +
            "  color = vec4(result.x, 0.0, 0.0, 0.0);\n" +
            "}";
    }

    _packFragShaderSource() {
        return this._shaderCommon() + "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc_st = (out_image_idx.y * out_image_num.x + out_image_idx.x) * 4;\n" +
            "\n" +
            "  vec4 data = texelFetch(input_texture, screen_coord, 0);\n" +
            "  vec4 bnp0 = texelFetch(param_texture, ivec2(oc_st, 0), 0);\n" +
            "  vec4 bnp1 = texelFetch(param_texture, ivec2(oc_st + 1, 0), 0);\n" +
            "  vec4 bnp2 = texelFetch(param_texture, ivec2(oc_st + 2, 0), 0);\n" +
            "  vec4 bnp3 = texelFetch(param_texture, ivec2(oc_st + 3, 0), 0);\n" +
            "\n" +
            "  color = vec4((data.x - bnp0.x) / sqrt(bnp0.y) * bnp0.z + bnp0.w,\n" +
            "               (data.y - bnp1.x) / sqrt(bnp1.y) * bnp1.z + bnp1.w,\n" +
            "               (data.z - bnp2.x) / sqrt(bnp2.y) * bnp2.z + bnp2.w,\n" +
            "               (data.w - bnp3.x) / sqrt(bnp3.y) * bnp3.z + bnp3.w);\n" +
            "}";
    }

    _createProgram() {
        let frag_source = null;
        if(this._packed) {
            frag_source = this._packFragShaderSource();
        } else {
            frag_source = this._fragShaderSource();
        }

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._param_texture_loc = this._gl.getUniformLocation(this._program, 'param_texture');
    }
}

export {BatchNorm}
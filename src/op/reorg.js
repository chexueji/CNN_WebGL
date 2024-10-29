import {createShader, createProgram} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class Reorg extends Operator {
    constructor(gl, packed) {
        super(gl, packed);

        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._input_texture_loc = 0;
        this._uniform_block_loc = 0;
        this._createProgram();

    }

    allocMemory(inputs) {
        let input = inputs[0];
        let input_storage = input.storage;
        let dims_in = input.dims();

        let channel_out = dims_in[0] * 4;

        let oh = parseInt(dims_in[1] / 2);
        let ow = parseInt(dims_in[2] / 2);

        let dims_out = Array.of(channel_out, oh, ow);

        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;

        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let reshape_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            dims_out[2], dims_out[1], // output width and height
            input_storage.blockHorizon, input_storage.blockVertical,
            dims_in[2], dims_in[1]]); // input width and height
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, reshape_param, this._gl.DYNAMIC_DRAW);
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
            console.log("Reorg(dim_in:" + inputs[0].dims() + ", dim_out:" + output[0].dims());
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

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size);\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int ic = oc / 4; // input channel\n" +
            "  int id_in_quad = oc % 4; // index in quad kernal(4 pixels)\n" +
            "\n" +
            "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size)) * ivec2(2, 2);\n" +
            "\n" +
            "  ivec2 in_image_idx = ivec2(ic % in_image_num.x, ic / in_image_num.x); // input image block index of h & w\n" +
            "\n" +
            "  ivec2 in_coord = ivec2(in_image_idx * in_image_size + image_coord_start) + ivec2(id_in_quad % 2,id_in_quad / 2); // h & w index\n" +
            "\n" +
            "  float cl = texelFetch(input_texture, in_coord, 0).x;\n" +
            "\n" +
            "  color = vec4(cl, 0, 0.0, 0.0);\n" +
            "}";
    }
    
    _packFragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "\n" +
            "  color = vec4(1.0, 0.0, 0.0, 0.0);\n" +
            "}";
    }



    _shaderCommon() {
        return "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "\n" +
            "uniform ShapeParam {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "\n" +
            "  ivec2 in_image_num;\n" +
            "  ivec2 in_image_size;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n";
    }

    _createProgram() {
        let frag_source = null;
        if (this._packed) {
            frag_source = this._packFragShaderSource();
        } else {
            frag_source = this._fragShaderSource();
        }

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ShapeParam');
    }

}

export {Reorg}
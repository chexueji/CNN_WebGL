import {createShader, createProgram} from "./common/common.js";
import {Tensor} from "./common/tensor.js"
import {Operator} from "./op/operator.js"
import {DEBUG} from "./common/common.js";

class RescaleImage extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._scale = 1;
        this._output_size = [];

        this._input_texture_loc = 0;

        this._createProgram();
    }

    allocMemory(inputs) {
      console.assert(false,'RescaleImage run allocMemory() of base class');
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

    get scale() {
        return this._scale;
    }

    set scale(value) {
        this._scale = value;
    }

    _createProgram() {
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, this._fragShaderSource());
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
            "out vec4 color;\n";
    }
}

class RescaleImage_Linear extends RescaleImage {
    constructor(gl, packed) {
        super(gl, packed);
    }

    allocMemory(inputs) {
        let input = inputs[0];
        let input_storage = input.storage;
        let dims_in = input.dims();//chw
        if (this._output_size.length == 0) {
            this._output_size = [dims_in[1] * this._upsample_scale,
                dims_in[2] * this._upsample_scale];
        }
        let dims_out = Array.of(dims_in[0], this._output_size[0], this._output_size[1]);
        //linear sampling
        let output_tensor = new Tensor(this._gl, this._packed, dims_out, null,null,2,true);

        return output_tensor;
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  color = texelFetch(input_texture, screen_coord, 0);\n" +
            "}";
    }
}

export {RescaleImage_Linear}
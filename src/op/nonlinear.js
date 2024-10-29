import {createShader, createProgram} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class Nonlinear extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._program = null;
        this._input_texture_loc = 0;
        this._createProgram();
    }

    allocMemory(inputs) {
        let input = inputs[0];
        //let input_storage = input.storage();
        let dims_in = input.dims();

        let output_tensor = new Tensor(this._gl, this._packed, dims_in);
        return output_tensor;
    }

    directRun(inputs) {
        let output_tensor = this.allocMemory(inputs);
        this.run(inputs, output_tensor);

        return output_tensor;
    }

    _postCreateProgram(frag_source) {
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
    }

    _createProgram() {
        console.error("Error: this function should be implemented in sub-class.");
    }
}

class Sigmoid extends Nonlinear {
    constructor(gl, packed) {
        super(gl, packed);
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

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("Sigmoid(input dims(" + input.dims() + "),output dims(" + output.dims() + "))")
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    _createProgram() {
        let frag_source = "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  float cl = 1.0 / (1.0 + exp(-texelFetch(input_texture, ivec2(gl_FragCoord.xy), 0).x));\n" +
            "  color = vec4(cl, 0.0, 0.0, 0.0);\n" +
            "}";

        this._postCreateProgram(frag_source);
    }
}

class Relu extends Nonlinear {
    constructor(gl, packed) {
        super(gl, packed);
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

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("Relu(input dims(" + input.dims() + "),output dims(" + output.dims() + "))");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    _createProgram() {
        let frag_source = "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = max(texelFetch(input_texture, ivec2(gl_FragCoord.xy), 0), 0.0);\n" +
            "}";

        this._postCreateProgram(frag_source);
    }
}

class HardTanh extends Nonlinear {
    constructor(gl, packed, max = 1, min = -1) {
        super(gl, packed);
        this._max = max;
        this._min = min;
        this._threshold_loc = 0;
        this._threshold_loc = this._gl.getUniformLocation(this._program, 'threshold');
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

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.uniform2f(this._threshold_loc, this.min, this.max);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("HardTanh(input dims(" + input.dims() + "),output dims(" + output.dims() + ") max val:" + this.max + " min val:" + this.min + ")");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    get max() {
        return this._max;
    }

    set max(value) {
        this._max = value;
    }

    get min() {
        return this._min;
    }

    set min(value) {
        this._min = value;
    }

    _createProgram() {
        let frag_source = "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform vec2 threshold;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = clamp(texelFetch(input_texture, ivec2(gl_FragCoord.xy), 0), threshold.x, threshold.y);\n" +
            "}";

        this._postCreateProgram(frag_source);

    }
}

export {Relu, HardTanh, Sigmoid}
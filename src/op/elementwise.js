import {createShader, createProgram} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class ElementWise extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._program = null;
        this._input_texture_loc = 0;
        this._input_other_loc = 0;

        this._createProgram();
    }

    allocMemory(inputs) {
        let input = inputs[0];
        let dims_in = input.dims();

        let output_tensor = new Tensor(this._gl, this._packed, dims_in);
        return output_tensor;
    }

    directRun(inputs) {
        let output_tensor = this.allocMemory(inputs);
        this.run(inputs, output_tensor);

        return output_tensor;
    }
}

class ElementWise_Add extends ElementWise {
    constructor(gl, packed) {
        super(gl, packed);
    }

    run(inputs, output, layerIndex) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        if (!input1) {
            console.error(this);
        }
        let in_storage0 = input0.storage;
        let in_storage1 = input1.storage;
        let out_storage = output.storage;

        output.bind();
        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);
        this._gl.uniform1i(this._input_other_loc, 1);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("ElementWise_Add(dims(" + input0.dims() + "))");
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
            "uniform sampler2D input_texture0;\n" +
            "uniform sampler2D input_texture1;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = texelFetch(input_texture0, ivec2(gl_FragCoord.xy), 0) + texelFetch(input_texture1, ivec2(gl_FragCoord.xy), 0);\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture0');
        this._input_other_loc = this._gl.getUniformLocation(this._program, 'input_texture1');
    }
}

class ElementWise_Avg extends ElementWise {
    constructor(gl, packed) {
        super(gl, packed);
    }

    run(inputs, output, layerIndex) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        if (!input1) {
            console.error(this);
        }
        let in_storage0 = input0.storage;
        let in_storage1 = input1.storage;
        let out_storage = output.storage;

        output.bind();
        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);
        this._gl.uniform1i(this._input_texture_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);
        this._gl.uniform1i(this._input_other_loc, 1);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("ElementWise_Avg(dims(" + input0.dims() + "))");
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
            "uniform sampler2D input_texture0;\n" +
            "uniform sampler2D input_texture1;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = texelFetch(input_texture0, ivec2(gl_FragCoord.xy), 0) + texelFetch(input_texture1, ivec2(gl_FragCoord.xy), 0);\n" +
            "  color /= 2.0;\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture0');
        this._input_other_loc = this._gl.getUniformLocation(this._program, 'input_texture1');
    }
}

class ElementWise_CAdd extends ElementWise {
    constructor(gl, packed, input_const = 0.0) {
        super(gl, packed);
        this._input_const = input_const;
    }

    get inputConst() {
        return this._input_const;
    }

    set inputConst(value) {
        this._input_const = value;
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
        this._gl.uniform1f(this._input_other_loc, this._input_const);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("ElementWise_CAdd(dims(" + input.dims() + "))");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform float input_const;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = texelFetch(input_texture, ivec2(gl_FragCoord.xy), 0) + input_const;\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._input_other_loc = this._gl.getUniformLocation(this._program, 'input_const');
    }
}


class ElementWise_CMul extends ElementWise {
    constructor(gl, packed, input_const = 1.0) {
        super(gl, packed);
        this._input_const = input_const;
    }

    get inputConst() {
        return this._input_const;
    }

    set inputConst(value) {
        this._input_const = value;
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
        this._gl.uniform1f(this._input_other_loc, this._input_const);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("ElementWise_CMul(dims(" + input.dims() + "))");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform float input_const;\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  color = texelFetch(input_texture, ivec2(gl_FragCoord.xy), 0) * input_const;\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._input_other_loc = this._gl.getUniformLocation(this._program, 'input_const');
    }
}

class ElementWise_SliceMul extends ElementWise {
    constructor(gl, packed) {
        super(gl, packed);
        this._uniform_block_loc = 0;
        this._uniform_binding_point = 0;
        this._uniform_buffer = 0;
    }

    allocMemory(inputs) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let dims_in0 = input0.dims();
        let dims_in1 = input1.dims();
        let input_storage0 = input0.storage;
        let input_storage1 = input1.storage;

        console.assert(dims_in0.length == 3 && dims_in1.length == 3,'dims mismatch');
        console.assert(dims_in0[0] == dims_in1[0] && dims_in1[1] == 1 && dims_in1[2] == 1,'SliceMul: the size of input0 and input1 mismatch');

        let output_tensor = new Tensor(this._gl, this._packed, dims_in0);
        // let output_storage = output_tensor.storage;
        // let dims_out = output_tensor.dims();

        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._uniform_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let reshape_param = new Int32Array([input_storage0.blockHorizon, input_storage0.blockVertical,
            dims_in0[2], dims_in0[1], // width and height of output is equal to input's respectively
            input_storage1.blockHorizon, input_storage1.blockVertical,
            dims_in1[2], dims_in1[1]]); // input1 width and height
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, reshape_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output, layerIndex) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let in_storage0 = input0.storage;
        let in_storage1 = input1.storage;
        let out_storage = output.storage;


        output.bind();
        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._uniform_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);

        this._gl.uniform1i(this._input_texture_loc, 0);
        this._gl.uniform1i(this._input_other_loc, 1);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("ElementWise_SliceMul(dims(" + input.dims() + "))");
            let result = output.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output.unbind();
    }

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input0_texture;\n" +
            "uniform sampler2D input1_texture;\n" +
            "\n" +
            "uniform ShapeParam {\n" +
            "  ivec2 in0_image_num;\n" +
            "  ivec2 in0_image_size;\n" +
            "\n" +
            "  ivec2 in1_image_num;\n" +
            "  ivec2 in1_image_size;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / in0_image_size);\n" +
            "  int oc = out_image_idx.y * in0_image_num.x + out_image_idx.x;\n" +
            "\n" +
            "  int ic = oc; // input channel\n" +
            "  ivec2 input1_coord = ivec2(ic % in1_image_num.x, ic / in1_image_num.x); \n" +
            "  ivec2 input0_coord = screen_coord; \n" +
            // "  ivec2 in_coord = ivec2(input_idx_in * in_image_size + image_coord_start); \n" +
            // "\n" +
            // "  ivec2 image_coord_start = (screen_coord - ivec2(out_image_idx * in0_image_size));\n" +
            // "\n" +
            // "  ivec2 in_image_idx = ivec2(ic % in_image_num.x, ic / in_image_num.x); // input image block index of h & w\n" +
            // "\n" +
            // "  ivec2 in_coord = ivec2(in_image_idx * in_image_size + image_coord_start) + ivec2(id_in_quad % 2,id_in_quad / 2); // h & w index\n" +
            "\n" +
            "  float cl = texelFetch(input0_texture, input0_coord, 0).x * texelFetch(input1_texture, input1_coord, 0).x;\n" +
            "\n" +
            "  color = vec4(cl, 0, 0.0, 0.0);\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input0_texture');
        this._input_other_loc = this._gl.getUniformLocation(this._program, 'input1_texture');
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ShapeParam');
    }
}

export {ElementWise_Add, ElementWise_CAdd, ElementWise_CMul, ElementWise_Avg, ElementWise_SliceMul};
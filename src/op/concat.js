import {createShader, createProgram, makeTypedTexture2D} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js";
import {Operator} from "./operator.js";
import {DEBUG} from "../common/common.js";

//Concat1D Op
class Concat1D extends Operator {
    constructor(gl, packed) {
        super(gl, packed);
        this._input_texture1_loc = 0;
        this._input_texture2_loc = 0;
        this._input_texture3_loc = 0;
        this._input_texture4_loc = 0;
        this._input_texture5_loc = 0;

        this._uniform_block_loc = 0;
        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._createProgram();
    }

    allocMemory(inputs) {


        let input0 = inputs[0];
        let dims_in0 = input0.dims();

        let dims_out;
        let width = dims_in0[2];
        let height = dims_in0[1];
        let channel = dims_in0[0];
        console.assert((channel == 1 && height == 1) || (channel == 1 && width == 1));
        let max_texture_size = this._gl.getParameter(this._gl.MAX_TEXTURE_SIZE);
        let height_size = 0
        let width_size = 0;
        let channel_concat = 0;
        if (channel == 1 && height == 1) {
            for (let idx = 0; idx < inputs.length; idx++) {
                let dims = inputs[idx].dims();
                console.assert(dims[0] == 1 && dims[1] == 1);
                width_size += dims[2];
            }
            dims_out = Array.of(1, 1, width_size);
            console.assert(max_texture_size >= width_size);
        } else {
            for (let idx = 0; idx < inputs.length; idx++) {
                let dims = inputs[idx].dims();
                console.assert(dims[0] == 1 && dims[2] == 1);
                height_size += dims[1];
            }
            console.assert(max_texture_size >= height_size);
            dims_out = Array.of(1, height_size, 1);
        }


        let factor = this._packed ? 4 : 1;
        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;
        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let concat_param = new Int32Array([inputs[0].dims()[2], inputs[0].dims()[1],
            inputs[1].dims()[2], inputs[1].dims()[1],//input width, height
            inputs[2].dims()[2], inputs[2].dims()[1],
            inputs[3].dims()[2], inputs[3].dims()[1],
            inputs[4].dims()[2], inputs[4].dims()[1], 0, 0, 0, 0, 0, 0]);
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, concat_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output, layerIndex) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let input2 = inputs[2];
        let input3 = inputs[3];
        let input4 = inputs[4];

        let in_storage0 = input0.storage;
        let in_storage1 = input1.storage;
        let in_storage2 = input2.storage;
        let in_storage3 = input3.storage;
        let in_storage4 = input4.storage;

        let out_storage = output.storage;

        output.bind();

        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
        // uniform
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._unifrom_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);
        this._gl.uniform1i(this._input_texture1_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);
        this._gl.uniform1i(this._input_texture2_loc, 1);

        this._gl.activeTexture(this._gl.TEXTURE2);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage2.texture);
        this._gl.uniform1i(this._input_texture3_loc, 2);

        this._gl.activeTexture(this._gl.TEXTURE3);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage3.texture);
        this._gl.uniform1i(this._input_texture4_loc, 3);

        this._gl.activeTexture(this._gl.TEXTURE4);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage4.texture);
        this._gl.uniform1i(this._input_texture5_loc, 4);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
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

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture1;\n" +
            "uniform sampler2D input_texture2;\n" +
            "uniform sampler2D input_texture3;\n" +
            "uniform sampler2D input_texture4;\n" +
            "uniform sampler2D input_texture5;\n" +
            "uniform ConcatParam {\n" +
            "  ivec4 in_image_size[4];\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 coord;\n" +
            "  if(screen_coord.x < in_image_size[0].x) {\n" +
            "      coord = ivec2(screen_coord.x,0);\n" +
            "      color = texelFetch(input_texture1, coord, 0);\n" +
            "  } else if(screen_coord.x < (in_image_size[0].x + in_image_size[0].z)) {\n" +
            "      coord = ivec2(screen_coord.x - in_image_size[0].x,0);\n" +
            "      color = texelFetch(input_texture2, coord, 0);\n" +
            "  } else if(screen_coord.x < (in_image_size[0].x + in_image_size[0].z + in_image_size[1].x)) {\n" +
            "      coord = ivec2(screen_coord.x - in_image_size[0].x - in_image_size[0].z,0);\n" +
            "      color = texelFetch(input_texture3, coord, 0);\n" +
            "  } else if(screen_coord.x < (in_image_size[0].x + in_image_size[0].z + in_image_size[1].x + in_image_size[1].z)) {\n" +
            "      coord = ivec2(screen_coord.x - in_image_size[0].x - in_image_size[0].z - in_image_size[1].x,0);\n" +
            "      color = texelFetch(input_texture4, coord, 0);\n" +
            "  } else if(screen_coord.x < (in_image_size[0].x + in_image_size[0].z + in_image_size[1].x + in_image_size[1].z + in_image_size[2].x)) {\n" +
            "      coord = ivec2(screen_coord.x - in_image_size[0].x - in_image_size[0].z - in_image_size[1].x - in_image_size[1].z,0);\n" +
            "      color = texelFetch(input_texture5, coord, 0);\n" +
            "  }\n" +
            "}";

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture1_loc = this._gl.getUniformLocation(this._program, 'input_texture1');
        this._input_texture2_loc = this._gl.getUniformLocation(this._program, 'input_texture2');
        this._input_texture3_loc = this._gl.getUniformLocation(this._program, 'input_texture3');
        this._input_texture4_loc = this._gl.getUniformLocation(this._program, 'input_texture4');
        this._input_texture5_loc = this._gl.getUniformLocation(this._program, 'input_texture5');

        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ConcatParam');
    }
}

class Concat2D_2 extends Operator {
    constructor(gl, packed, concat_dim) {
        super(gl, packed);
        this._input_texture1_loc = 0;
        this._input_texture2_loc = 0;
        this._concat_dim = concat_dim;

        this._uniform_block_loc = 0;
        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._createProgram();
    }

    allocMemory(inputs) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let dims_in0 = input0.dims();
        let dims_in1 = input1.dims();
        let dims_out = Array.of(dims_in0[0] + dims_in1[0], dims_in0[1], dims_in0[2]);

        console.assert(inputs.length == 2,'Concat2D: num of input nodes is larger than 2');
        console.assert(this._concat_dim < 2,'Concat2D: concat_dim 2 is Not supported');
        console.assert(dims_in0[1] == dims_in1[1] && dims_in0[2] == dims_in1[2],'Concat2D: dims mismatch in inputs');

        let factor = this._packed ? 4 : 1;
        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;
        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let concat_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            dims_in0[2], dims_in0[1],//input width, height
            dims_in0[0] / factor, dims_in1[0] / factor,  // channel_input
            input0.storage.blockHorizon, input0.storage.blockVertical,
            input1.storage.blockHorizon, input1.storage.blockVertical, 0, 0]);
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, concat_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output) {
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
        // uniform
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._unifrom_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);
        this._gl.uniform1i(this._input_texture1_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);
        this._gl.uniform1i(this._input_texture2_loc, 1);

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

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture1;\n" +
            "uniform sampler2D input_texture2;\n" +
            "uniform ConcatParam {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "  ivec2 in_channels;\n" +
            "  ivec2 in_image_num1;\n" +
            "  ivec2 in_image_num2;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image_size);\n" +
            "\n" +
            "  if(oc < in_channels.x) {\n" +
            "      ivec2 in_image_idx = ivec2(oc % in_image_num1.x, oc / in_image_num1.x);\n" +
            "      ivec2 coord = in_image_idx * out_image_size + image_coord;\n" +
            "      color = texelFetch(input_texture1, coord, 0);\n" +
            "  } else {\n" +
            "      ivec2 in_image_idx = ivec2((oc - in_channels.x) % in_image_num2.x, (oc - in_channels.x) / in_image_num2.x);\n" +
            "      ivec2 coord = in_image_idx * out_image_size + image_coord;\n" +
            "      color = texelFetch(input_texture2, coord, 0);\n" +
            "  }\n" +
            "}";
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture1_loc = this._gl.getUniformLocation(this._program, 'input_texture1');
        this._input_texture2_loc = this._gl.getUniformLocation(this._program, 'input_texture2');
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ConcatParam');
    }

    get concat_dim()
    {
        return this._concat_dim;
    }

    set concat_dim(concat_dim)
    {
        this._concat_dim = concat_dim;
    }
}


class Concat2D_3 extends Operator {
    constructor(gl, packed, concat_dim) {
        super(gl, packed);
        this._input_texture1_loc = 0;
        this._input_texture2_loc = 0;
        this._input_texture3_loc = 0;
        this._concat_dim = concat_dim;

        this._uniform_block_loc = 0;
        this._uniform_buffer = null;
        this._unifrom_binding_point = 0;
        this._createProgram();
    }

    allocMemory(inputs) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let input2 = inputs[2];
        let dims_in0 = input0.dims();
        let dims_in1 = input1.dims();
        let dims_in2 = input2.dims();
        let dims_out = Array.of(dims_in0[0] + dims_in1[0] + dims_in2[0], dims_in0[1], dims_in0[2]);

        console.assert(inputs.length == 3,'Concat2D: num of input nodes is larger than 3');
        console.assert(this._concat_dim < 2,'Concat2D: concat_dim 2 is Not supported');
        console.assert(dims_in0[1] == dims_in1[1] && dims_in0[2] == dims_in1[2] && dims_in0[1] == dims_in2[1] && dims_in0[2] == dims_in2[2],'Concat2D: dims mismatch in inputs');

        let factor = this._packed ? 4 : 1;
        let output_tensor = new Tensor(this._gl, this._packed, dims_out);
        let output_storage = output_tensor.storage;
        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let concat_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            dims_in0[2], dims_in0[1],//input width, height
            dims_in0[0] / factor, dims_in1[0] / factor,  // channel_input
            input0.storage.blockHorizon, input0.storage.blockVertical,
            input1.storage.blockHorizon, input1.storage.blockVertical,
            input2.storage.blockHorizon, input2.storage.blockVertical]);
        this._gl.bufferData(this._gl.UNIFORM_BUFFER, concat_param, this._gl.DYNAMIC_DRAW);
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, null);

        return output_tensor;
    }

    run(inputs, output) {
        let input0 = inputs[0];
        let input1 = inputs[1];
        let input2 = inputs[2];
        let in_storage0 = input0.storage;
        let in_storage1 = input1.storage;
        let in_storage2 = input2.storage;
        let out_storage = output.storage;

        output.bind();

        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);
        // uniform
        this._gl.bindBufferBase(this._gl.UNIFORM_BUFFER, this._unifrom_binding_point, this._uniform_buffer);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage0.texture);
        this._gl.uniform1i(this._input_texture1_loc, 0);

        this._gl.activeTexture(this._gl.TEXTURE1);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage1.texture);
        this._gl.uniform1i(this._input_texture2_loc, 1);

        this._gl.activeTexture(this._gl.TEXTURE2);
        this._gl.bindTexture(this._gl.TEXTURE_2D, in_storage2.texture);
        this._gl.uniform1i(this._input_texture3_loc, 2);

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

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "\n" +
            "uniform sampler2D input_texture1;\n" +
            "uniform sampler2D input_texture2;\n" +
            "uniform sampler2D input_texture3;\n" +
            "uniform ConcatParam {\n" +
            "  ivec2 out_image_num;\n" +
            "  ivec2 out_image_size;\n" +
            "  ivec2 in_channels;\n" +
            "  ivec2 in_image_num1;\n" +
            "  ivec2 in_image_num2;\n" +
            "  ivec2 in_image_num3;\n" +
            "};\n" +
            "\n" +
            "out vec4 color;\n" +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image_size);\n" +
            "\n" +
            "  if(oc < in_channels.x) {\n" +
            "      ivec2 in_image_idx = ivec2(oc % in_image_num1.x, oc / in_image_num1.x);\n" +
            "      ivec2 coord = in_image_idx * out_image_size + image_coord;\n" +
            "      color = texelFetch(input_texture1, coord, 0);\n" +
            "  } else if(oc < (in_channels.x + in_channels.y)) {\n" +
            "      ivec2 in_image_idx = ivec2((oc - in_channels.x) % in_image_num2.x, (oc - in_channels.x) / in_image_num2.x);\n" +
            "      ivec2 coord = in_image_idx * out_image_size + image_coord;\n" +
            "      color = texelFetch(input_texture2, coord, 0);\n" +
            "  } else {\n" +
            "      ivec2 in_image_idx = ivec2((oc - in_channels.y - in_channels.x) % in_image_num3.x, (oc - in_channels.y - in_channels.x) / in_image_num3.x);\n" +
            "      ivec2 coord = in_image_idx * out_image_size + image_coord;\n" +
            "      color = texelFetch(input_texture3, coord, 0);\n" +
            "  }\n" +
            "}";
        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture1_loc = this._gl.getUniformLocation(this._program, 'input_texture1');
        this._input_texture2_loc = this._gl.getUniformLocation(this._program, 'input_texture2');
        this._input_texture3_loc = this._gl.getUniformLocation(this._program, 'input_texture3');
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ConcatParam');
    }

    get concat_dim()
    {
        return this._concat_dim;
    }

    set concat_dim(concat_dim)
    {
        this._concat_dim = concat_dim;
    }
}

export {Concat2D_2, Concat2D_3, Concat1D};
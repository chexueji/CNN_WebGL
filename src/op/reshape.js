import {createShader, createProgram} from "../common/common.js";
import {Tensor, Storage} from "../common/tensor.js"
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

class Reshape extends Operator {
    constructor(gl, packed, dims_out) {
        super(gl, packed);

        this._dims_out = dims_out;
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

        this._checkDims(dims_in, this._dims_out);

        let output_tensor = new Tensor(this._gl, this._packed, this._dims_out);
        let output_storage = output_tensor.storage;

        // uniform binding
        this._gl.uniformBlockBinding(this._program, this._uniform_block_loc, this._unifrom_binding_point);
        this._uniform_buffer = this._gl.createBuffer();
        this._gl.bindBuffer(this._gl.UNIFORM_BUFFER, this._uniform_buffer);
        let reshape_param = new Int32Array([output_storage.blockHorizon, output_storage.blockVertical,
            this.dims_out[2], this.dims_out[1], // output width and height
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
            console.log("Reshape(dim_in:" + inputs[0].dims() + ", dim_out:" + output.dims());
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
        this._uniform_block_loc = this._gl.getUniformBlockIndex(this._program, 'ShapeParam');
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

    get dims_out() {
        return this._dims_out;
    }

    set dims_out(dims_out) {
        this._dims_out = dims_out;
    }

}

class View extends Reshape {
    constructor(gl, packed, dims_out) {
        super(gl, packed, dims_out);
    }

    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size);\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "  ivec2 out_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size));\n" +
            "  int image_global_idx = oc*(out_image_size.x*out_image_size.y)+(out_coord_start.y*out_image_size.x)+out_coord_start.x;\n" +
            "  int ic = image_global_idx/(in_image_size.x*in_image_size.y);\n" +
            "  int image_idx = image_global_idx%(in_image_size.x*in_image_size.y);\n" +
            "  ivec2 image_coord = ivec2(image_idx%in_image_size.x,image_idx/in_image_size.x);\n" +
            "  ivec2 in_image_idx = ivec2(ic % in_image_num.x, ic / in_image_num.x);\n" +
            "  ivec2 in_image_coord = ivec2(in_image_idx * in_image_size + image_coord);\n" +
            "\n" +
            "  color = vec4(texelFetch(input_texture, in_image_coord, 0).x, 0.0, 0.0, 0.0);\n" +
            "}";
    }
    //to do
    _packFragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "\n" +
            "  color = vec4(0.0, 0.0, 0.0, 0.0);\n" +
            "}";
    }

    _checkDims(dims_in, dims_out) {
        let num_in = 1;
        dims_in.forEach(function (v) {
            num_in *= v;
        });
        let num_out = 1;
        let neg_count = 0;
        let neg_index = 0;
        try {
            dims_out.forEach(function (v, index) {
                if (v === -1) {
                    v = -v;
                    neg_count++;
                    neg_index = index;
                }
                num_out *= v;
            });
        } catch (e) {
            console.log(e.message);
        }

        if (neg_count == 0 && num_in != num_out) {
            console.error("Error: reshape size mismatch original size.");
            return;
        }

        if (neg_count == 1) {
            dims_out[neg_index] = num_in / num_out;
        } else if (neg_count >= 2) {
            console.error("Error: reshape size invalidate.");
            return;
        }

    }
}

class Permute extends Reshape {
    constructor(gl, packed, dims_out) {
        super(gl, packed, dims_out);
    }

    // chw 2 hwc
    _fragShaderSource() {
        return this._shaderCommon() +
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image_size); // channel\n" +
            "  int oc = out_image_idx.y * out_image_num.x + out_image_idx.x;\n" +
            "  ivec2 out_coord_start = (screen_coord - ivec2(out_image_idx * out_image_size));\n" +
            "\n" +
            "  int image_global_idx = out_coord_start.x * (out_image_num.x * out_image_num.y) * out_image_size.y + oc*out_image_size.y + out_coord_start.y;\n" +
            "\n" +
            "  int ic = image_global_idx / (in_image_size.x*in_image_size.y); // input channel\n" +
            "  int image_idx = image_global_idx % (in_image_size.x*in_image_size.y);\n" +
            "\n" +
            "  ivec2 image_coord_start = ivec2(image_idx%in_image_size.x,image_idx/in_image_size.x);// input width,height\n" +
            "\n" +
            "  ivec2 in_image_idx = ivec2(ic % in_image_num.x, ic / in_image_num.x);\n" +
            "  ivec2 in_coord = ivec2(in_image_idx * in_image_size + image_coord_start);\n" +
            "\n" +
            "  color = vec4(texelFetch(input_texture, in_coord, 0).x, 0.0, 0.0, 0.0);\n" +
            "\n" +
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

    _checkDims(dims_in, dims_out) {
        //CHW 2 HWC
        try {
            if (dims_in.length == 3) {
                //CHW to HWC
                dims_out[0] = dims_in[1];
                dims_out[1] = dims_in[2];
                dims_out[2] = dims_in[0];
            } else {
                console.error("Permute Op dim size:" + dims_out.toString() + "not implemented,CHW to HWC only");
            }
        } catch (error) {
            console.error(error.message);
        }

    }
}


export {View, Permute}
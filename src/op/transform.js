import {createShader, createProgram} from "../common/common.js";
import {Operator} from "./operator.js"
import {DEBUG} from "../common/common.js";

/**
 * From RGB(A) to R32F
 */
class Transform extends Operator {
    constructor(gl, packed, scale = 1.0, trans = 0.0) {
        super(gl, packed);
        this._scale = scale;
        this._trans = trans;
        this._input_texture_loc = 0;
        this._out_param_loc = 0;
        this._trans_param_loc = 0;
        this._createProgram();
    }

    run(input_texture, output_tensor, layerIndex) {
        let out_storage = output_tensor.storage;
        let dims_out = output_tensor.dims();

        output_tensor.bind();
        this._gl.useProgram(this._program);
        let vertexPosLocation = 0; // set with GLSL layout qualifier
        this._gl.enableVertexAttribArray(vertexPosLocation);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, this._vertex_buffer);
        this._gl.vertexAttribPointer(vertexPosLocation, 2, this._gl.FLOAT, false, 0, 0);
        this._gl.bindBuffer(this._gl.ARRAY_BUFFER, null);

        this._gl.activeTexture(this._gl.TEXTURE0);
        this._gl.bindTexture(this._gl.TEXTURE_2D, input_texture);
        this._gl.uniform1i(this._input_texture_loc, 0);
        this._gl.uniform4i(this._out_param_loc, out_storage.blockHorizon, out_storage.blockVertical, dims_out[2], dims_out[1]);
        this._gl.uniform2f(this._trans_param_loc, this._scale, this._trans);

        this._gl.viewport(0, 0, out_storage.textureWidth, out_storage.textureHeight);
        this._gl.clearColor(1.0, 1.0, 1.0, 1.0);
        this._gl.clear(this._gl.COLOR_BUFFER_BIT);
        this._gl.drawArrays(this._gl.TRIANGLES, 0, 6);
        this._gl.useProgram(null);

        if (DEBUG) {
            console.log("transform data:");
            let result = output_tensor.rawData();
            console.log(result);
            console.save(result, layerIndex + "_output.json");
        }

        output_tensor.unbind();
    }

    _createProgram() {
        let frag_source =
            "#version 300 es\n" +
            "precision highp float;\n" +
            "precision highp int;\n" +
            "precision highp sampler2D;\n" +
            "uniform sampler2D input_texture;\n" +
            "uniform ivec4 out_image;\n" +
            "uniform vec2 transforms;\n" +
            "out vec4 color;\n";

        if (this._packed) {
            frag_source +=
                "void main() {\n" +
                "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
                "  vec4 value = texelFetch(input_texture, screen_coord, 0) * transforms.x + transforms.y;" +
                "  color = vec4(value.xyz, 0.0);\n" +
                "}";
        } else {
            frag_source +=
                // "void main() {\n" +
                // "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
                // "  ivec2 out_image_idx = ivec2(screen_coord / out_image.zw);\n" +
                // "  int channel = out_image_idx.y * out_image.x + out_image_idx.x;\n" +
                // "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image.zw);\n" +
                // "  color = vec4(texelFetch(input_texture, image_coord, 0)[channel]*transforms.x, 0.0, 0.0, 0.0);\n" +
                // "}";
            //RGB 2 YUV
            "void main() {\n" +
            "  ivec2 screen_coord = ivec2(gl_FragCoord.xy);\n" +
            "  ivec2 out_image_idx = ivec2(screen_coord / out_image.zw);\n" +
            "  int channel = out_image_idx.y * out_image.x + out_image_idx.x;\n" +
            "  ivec2 image_coord = screen_coord - ivec2(out_image_idx * out_image.zw);\n" +
            " \n" +
            "  vec4 color_rgba = texture(input_texture, vec2(image_coord.xy)/vec2(out_image.zw));\n"+
                // "  vec4 color_rgba = texelFetch(input_texture, image_coord, 0);\n"+
            "  float y = 0.299 * color_rgba.r + 0.587 * color_rgba.g + 0.114 * color_rgba.b;\n" +
            "  float u = 0.492 * (color_rgba.b - y) + 0.5;\n" +
            "  float v  = 0.877 * (color_rgba.r - y) + 0.5;\n" +
            "  vec4 color_yuv = vec4(y,u,v,color_rgba.a);\n" +
            "\n" +
            "  color = vec4(color_yuv[channel]*transforms.x, 0.0, 0.0, 0.0);\n" +
            "}";
        }

        let frag_shader = createShader(this._gl, this._gl.FRAGMENT_SHADER, frag_source);
        this._program = createProgram(this._gl, this._vert_shader, frag_shader);

        this._input_texture_loc = this._gl.getUniformLocation(this._program, 'input_texture');
        this._out_param_loc = this._gl.getUniformLocation(this._program, 'out_image');
        this._trans_param_loc = this._gl.getUniformLocation(this._program, 'transforms');
    }
}

export {Transform};
package org.tensorflow.lite.examples.detection;

import android.content.Context;

import java.util.ArrayList;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Model {

    protected final String checkpoint;
    protected ModelFactory.GeneralModel generalModel;
    protected String name;
    protected TensorFlowInferenceInterface inferenceEngine;
    protected HashMap<String, float[]> results;

    public Model(Context context, ModelFactory.GeneralModel generalModel, String name, String checkpoint){
        this.generalModel = generalModel;
        this.name = name;
        this.checkpoint = checkpoint;
        this.inferenceEngine = new TensorFlowInferenceInterface(context.getAssets(), checkpoint);
        this.results = new HashMap<>();

    }

    public float[] doInference(float[] input, int width, int height){
        float[] output = new float[height*width];

        this.inferenceEngine.feed(
                "im0:0", input, 1,
                height, width, 3);
        this.inferenceEngine.run(new String[]{"PSD/resize_images/ResizeBilinear:0"});
        this.inferenceEngine.fetch("PSD/resize_images/ResizeBilinear:0", output);
        return output;
    }

}

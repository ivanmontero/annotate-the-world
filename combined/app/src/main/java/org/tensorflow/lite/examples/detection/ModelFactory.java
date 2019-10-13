package org.tensorflow.lite.examples.detection;

import android.content.Context;

import java.util.ArrayList;
import java.util.List;

public class ModelFactory {
    private final Context context;

    private List<Model> models;

    public enum GeneralModel{
        PYDNET_PP
    }

    public ModelFactory(Context context){
        this.context = context;
        this.models = new ArrayList<>();
        models.add(createPydnetPP());
    }

    public Model getModel(int index ){
        return models.get(index);
    }

    private Model createPydnetPP(){
        Model pydnetPP;
        pydnetPP = new Model(context, GeneralModel.PYDNET_PP, "Pydnet++", "file:///android_asset/optimized_pydnet++.pb");
        return pydnetPP;
    }
}

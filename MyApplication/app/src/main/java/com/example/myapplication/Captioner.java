package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import org.json.*;

import org.pytorch.Tensor;
import org.pytorch.Module;
import org.pytorch.IValue;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Dictionary;
import java.util.List;

public class Captioner {
    Module encoder;
    Module decoder;


    float[] mean = {0.485f, 0.456f, 0.406f};
    float[] std = {0.229f, 0.224f, 0.225f};

    public Captioner(String encoderPath, String decoderPath){
        encoder = Module.load(encoderPath);
        decoder = Module.load(decoderPath);
    }


    public Tensor preprocess(Bitmap bitmap, int size){
        bitmap = Bitmap.createScaledBitmap(bitmap, size, size, false);

        return TensorImageUtils.bitmapToFloat32Tensor(bitmap,this.mean,this.std);
    }

    public Tensor get_features(Bitmap bitmap){

        Tensor tensor = preprocess(bitmap,224);
        tensor.shape();
        IValue inputs = IValue.from(tensor);
        Tensor features = encoder.forward(inputs).toTensor();

        System.out.println(features);
        return features;
    }

    public String generate_caption(Tensor features, Context context) throws JSONException {

        Tensor[] output = decoder.forward(IValue.from(features)).toTensorList();
        Tensor[] o = IValue.listFrom(output).toTensorList();

        List liO = Arrays.asList(output);

        String[] caption = new String[20];

        String word2id = loadJSONFromAsset(context);
        JSONObject w2id = new JSONObject(word2id);

        Integer tmp = 0;
        for (int i = 0; i < liO.size(); i++){
            Tensor x = (Tensor) liO.get(i);
            String word_id = x.toString();


            word_id = word_id.replaceAll("\\D+","");
            System.out.println(word_id);

            String[] parts = word_id.split("32");
            String id = parts[0];
            System.out.println(id);

            String word = w2id.getString(id);

            if (!word.equals("<SOS>") && !word.equals("null") && !word.equals("<EOS>")){
                caption[tmp] = word;
                tmp++;
            }


        }


        System.out.println(caption);
        StringBuilder caps = new StringBuilder();

        for (int i = 0; i < caption.length; i++) {
            caps.append(caption[i]).append(" ");
        }
        caps.toString().replace("null", "");
        return caps.toString();

    }

    public String loadJSONFromAsset(Context context) {
        String json = null;
        try {
            InputStream is = context.getAssets().open("fixed_word_to_id.json");

            int size = is.available();

            byte[] buffer = new byte[size];

            is.read(buffer);

            is.close();

            json = new String(buffer, "UTF-8");


        } catch (IOException ex) {
            ex.printStackTrace();
            return null;
        }
        return json;

    }
}

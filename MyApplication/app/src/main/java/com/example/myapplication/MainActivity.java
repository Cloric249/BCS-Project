package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.provider.Settings;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.ListView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONException;
import org.pytorch.Tensor;

import wseemann.media.FFmpegMediaMetadataRetriever;

import static com.example.myapplication.Utils.assetFilePath;

public class MainActivity extends AppCompatActivity {

    Context context;
    Captioner captioner;
    private ImageView imageView;
    private ImageView frame1;
    private ImageView frame2;
    private ImageView frame3;
    private ImageView frame4;
    private ImageView frame5;
    private ImageView frame6;
    private ImageView frame7;
    private ImageView frame8;
    private ImageView frame9;
    private ImageView frame10;
    private ImageView frame11;
    private ImageView frame12;
    private ListView listView;
    private Bitmap[] bmp;
    private int mode;
    public static final int PERMISSION_EXTERNAL_STORAGE = 1;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        context = this;
        captioner = new Captioner(assetFilePath(this, "Encoder.pt"),
                assetFilePath(this, "Decoder.pt"));


        imageView = (ImageView) findViewById(R.id.image);
        frame1 = (ImageView) findViewById(R.id.frame1);
        frame2 = (ImageView) findViewById(R.id.frame2);
        frame3 = (ImageView) findViewById(R.id.frame3);
        frame4 = (ImageView) findViewById(R.id.frame4);
        frame5 = (ImageView) findViewById(R.id.frame5);
        frame6 = (ImageView) findViewById(R.id.frame6);
        frame7 = (ImageView) findViewById(R.id.frame7);
        frame8 = (ImageView) findViewById(R.id.frame8);
        frame9 = (ImageView) findViewById(R.id.frame9);
        frame10 = (ImageView) findViewById(R.id.frame10);
        frame11 = (ImageView) findViewById(R.id.frame11);
        frame12 = (ImageView) findViewById(R.id.frame12);
        Button button = (Button) findViewById(R.id.button_id);
        Button button_image = (Button) findViewById(R.id.get_image);
        Button button_video = (Button) findViewById(R.id.get_Video);

        String[] array = {"Caption 1: ", "Caption 2: ", "Caption 3: ", "Caption 4: ", "Caption 5: "
        ,"Caption 6: ","Caption 7: ","Caption 8: ","Caption 9: ","Caption 10: ","Caption 11: ","Caption 12: "};

        ArrayAdapter adapter = new ArrayAdapter<String>(this,R.layout.res_layout, array);
        listView = (ListView) findViewById(R.id.list);
        listView.setAdapter(adapter);

        button_image.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imageView.setImageDrawable(null);
                frame1.setImageDrawable(null);
                frame2.setImageDrawable(null);
                frame3.setImageDrawable(null);
                frame4.setImageDrawable(null);
                frame5.setImageDrawable(null);
                frame6.setImageDrawable(null);
                frame7.setImageDrawable(null);
                frame8.setImageDrawable(null);
                frame9.setImageDrawable(null);
                frame10.setImageDrawable(null);
                frame11.setImageDrawable(null);
                frame12.setImageDrawable(null);
                selectImage(context);
                mode = 0;

            }
        });


        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (mode == 0) {
                    BitmapDrawable drawable = (BitmapDrawable) imageView.getDrawable();
                    if (drawable != null) {
                        Bitmap bitmap = drawable.getBitmap();
                        Tensor features = captioner.get_features(bitmap);
                        String output = null;
                        try {
                            output = captioner.generate_caption(features, context);
                            output = output.replace("null", "");
                            array[0] = ("Caption " + String.valueOf(1) + ": " + output);
                            adapter.notifyDataSetChanged();
                        } catch (JSONException e) {
                            e.printStackTrace();
                        }
                        System.out.println(output);
                    }

                }
                else if (mode == 1) {
                    for (int i = 0; i < bmp.length; i++){
                        Bitmap bitmap = bmp[i];
                        if (bitmap != null) {
                            Tensor features = captioner.get_features(bitmap);
                            String output = null;
                            try {
                                output = captioner.generate_caption(features, context);
                                output = output.replace("null", "");
                                array[i] = ("Caption " + String.valueOf(i+1) + ": " + output);
                                adapter.notifyDataSetChanged();
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
            }
        });

        button_video.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                imageView.setImageDrawable(null);
                frame1.setImageDrawable(null);
                frame2.setImageDrawable(null);
                frame3.setImageDrawable(null);
                frame4.setImageDrawable(null);
                frame5.setImageDrawable(null);
                frame6.setImageDrawable(null);
                frame7.setImageDrawable(null);
                frame8.setImageDrawable(null);
                frame9.setImageDrawable(null);
                frame10.setImageDrawable(null);
                frame11.setImageDrawable(null);
                frame12.setImageDrawable(null);

                selectVideo(context);
                mode = 1;
            }
        });
    }

    private void selectVideo(Context context) {
        final CharSequence[] options = {"Choose from Gallery","Cancel" };
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_EXTERNAL_STORAGE);
        }
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Choose A Video");

        builder.setItems(options, new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int item) {
                if (options[item].equals("Choose from Gallery")) {
                    Intent pickVideo = new Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(pickVideo , 2);

                } else if (options[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }



    private void selectImage(Context context) {
        final CharSequence[] options = { "Take Photo", "Choose from Gallery","Cancel" };
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M
                && ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE},
                    PERMISSION_EXTERNAL_STORAGE);
        }
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle("Choose your profile picture");

        builder.setItems(options, new DialogInterface.OnClickListener() {

            @Override
            public void onClick(DialogInterface dialog, int item) {


                if (options[item].equals("Take Photo")) {
                    Intent takePicture = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(takePicture, 0);

                } else if (options[item].equals("Choose from Gallery")) {
                    Intent pickPhoto = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(pickPhoto , 1);

                } else if (options[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_CANCELED) {
            switch (requestCode) {
                case 0:
                    if (resultCode == RESULT_OK && data != null) {
                        Bitmap selectedImage = (Bitmap) data.getExtras().get("data");
                        imageView.setImageBitmap(selectedImage);

                    }

                    break;
                case 1:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedImage = data.getData();
                        String[] filePathColumn = {MediaStore.Images.Media.DATA};
                        if (selectedImage != null) {
                            Cursor cursor = getContentResolver().query(selectedImage,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();

                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String picturePath = cursor.getString(columnIndex);
                                imageView.setImageBitmap(BitmapFactory.decodeFile(picturePath));
                                cursor.close();
                            }
                        }

                    }

                case 2:
                    if (resultCode == RESULT_OK && data != null) {
                        Uri selectedVideo = data.getData();
                        String[] filePathColumn = {MediaStore.Video.Media.DATA};
                        if (selectedVideo != null) {
                            Cursor cursor = getContentResolver().query(selectedVideo,
                                    filePathColumn, null, null, null);
                            if (cursor != null) {
                                cursor.moveToFirst();

                                int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
                                String videoPath = cursor.getString(columnIndex);
                                extractKeyFrames(videoPath);
                                cursor.close();
                            }
                        }

                    }
                    break;
            }
        }
    }





        @Override
        public void onRequestPermissionsResult(final int requestCode, @NonNull final String[] permissions, @NonNull final int[] grantResults) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            if (requestCode == PERMISSION_EXTERNAL_STORAGE) {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission granted.
                } else {
                    // User refused to grant permission.
                }
            }
        }




    public  boolean isStoragePermissionGranted() {
        if (Build.VERSION.SDK_INT >= 23) {

            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {

                return true;
            } else {

                ActivityCompat.requestPermissions(this, new String[]{ Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
                return false;
            }
        }
        else { //permission is automatically granted on sdk<23 upon installation
            return true;
        }
    }

    public void extractKeyFrames(String videoPath) {
        FFmpegMediaMetadataRetriever med = new FFmpegMediaMetadataRetriever();
        med.setDataSource(videoPath);

        String time = med.extractMetadata(FFmpegMediaMetadataRetriever.METADATA_KEY_DURATION);
        int iTime = Integer.parseInt(time)/1000;
        int extractEvery;
        int frameTime = 0;
        int imgView = 0;
        bmp = new Bitmap[12];
        ImageView frame = null;
        if (iTime < 10) {
            extractEvery = 2;
            for (int i = 0; i < extractEvery; i++) {
                frameTime = frameTime + extractEvery;
                imgView = imgView + 1;
                if (imgView == 1) {
                   frame = frame1;
                }
                if (imgView == 2) {
                    frame = frame2;
                }
                if (imgView == 3) {
                    frame = frame3;
                }
                if (imgView == 4) {
                    frame = frame4;
                }
                if (imgView == 5) {
                    frame = frame5;
                }

                Bitmap img = med.getFrameAtTime((frameTime*1000000) , FFmpegMediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                bmp[i] = img;
                frame.setImageBitmap(img);
            }
        }
        else if (10 <= iTime && iTime < 30) {
            extractEvery = 5;
            for (int i = 0; i < 6; i++) {
                frameTime = frameTime + extractEvery;
                imgView = imgView + 1;
                if (imgView == 1) {
                    frame = frame1;
                }
                if (imgView == 2) {
                    frame = frame2;
                }
                if (imgView == 3) {
                    frame = frame3;
                }
                if (imgView == 4) {
                    frame = frame4;
                }
                if (imgView == 5) {
                    frame = frame5;
                }
                if (imgView == 6) {
                    frame = frame6;
                }

                Bitmap img = med.getFrameAtTime((frameTime*1000000), FFmpegMediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                bmp[i] = img;
                frame.setImageBitmap(img);
            }
        }

        else if (30 <= iTime && iTime < 60) {
            extractEvery = 6;
            for (int i = 0; i < 10; i++) {
                frameTime = frameTime + extractEvery;
                imgView = imgView + 1;
                if (imgView == 1) {
                    frame = frame1;
                }
                if (imgView == 2) {
                    frame = frame2;
                }
                if (imgView == 3) {
                    frame = frame3;
                }
                if (imgView == 4) {
                    frame = frame4;
                }
                if (imgView == 5) {
                    frame = frame5;
                }
                if (imgView == 6) {
                    frame = frame6;
                }
                if (imgView == 7) {
                    frame = frame7;
                }
                if (imgView == 8) {
                    frame = frame8;
                }
                if (imgView == 9) {
                    frame = frame9;
                }
                if (imgView == 10) {
                    frame = frame10;
                }

                Bitmap img = med.getFrameAtTime((frameTime*1000000), FFmpegMediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                bmp[i] = img;
                frame.setImageBitmap(img);
            }
        }

        else {
            extractEvery = iTime/12;
            for (int i = 0; i < 12; i++) {
                frameTime = frameTime + extractEvery;
                imgView = imgView + 1;
                if (imgView == 1) {
                    frame = frame1;
                }
                if (imgView == 2) {
                    frame = frame2;
                }
                if (imgView == 3) {
                    frame = frame3;
                }
                if (imgView == 4) {
                    frame = frame4;
                }
                if (imgView == 5) {
                    frame = frame5;
                }
                if (imgView == 6) {
                    frame = frame6;
                }
                if (imgView == 7) {
                    frame = frame7;
                }
                if (imgView == 8) {
                    frame = frame8;
                }
                if (imgView == 9) {
                    frame = frame9;
                }
                if (imgView == 10) {
                    frame = frame10;
                }
                if (imgView == 11) {
                    frame = frame11;
                }
                if (imgView == 12) {
                    frame = frame12;
                }

                Bitmap img = med.getFrameAtTime((frameTime*1000000), FFmpegMediaMetadataRetriever.OPTION_CLOSEST_SYNC);
                bmp[i] = img;
                frame.setImageBitmap(img);
            }
        }
        med.release();
    }
}






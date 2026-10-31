import threading  # スレッド処理用
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance,ImageDraw
import cv2
import matplotlib.pyplot as plt
import math
import streamlit as st

# ==========================================
#  画像加工用の関数（フィルター）
# ==========================================


def load_input_image(uploaded_file):
    #アップロードされたファイルを１枚のPIL画像として準備する関数
    if uploaded_file is None:
        return None
    
    #アップロードファイルがリストで渡された場合の安全対策
    if isinstance(uploaded_file, list):
        uploaded_file=uploaded_file[0]

    #１枚のPIL画像として読み込む
    images=Image.open(uploaded_file)
    return images

def apply_fov_mask(images, fov_degrees, max_fov=360):
    """
    画像に対して指定された視野角(fov_degrees)以外の左右領域を黒くマスクする
    """
    if fov_degrees>=max_fov:
        return images #ほぼ全集見える場合は加工なし
    width, height=images.size

    #見えている割合を計算(例：犬 250° / 360° =約 0.69)
    visible_ratio=fov_degrees / max_fov

    #左右それぞれのカット(黒塗り)幅を計算
    hiden_ratio_one_side=(1.0 - visible_ratio) / 2.0
    mask_width=int(width * hiden_ratio_one_side)

    #画像のコピーを作成して黒枠を描画
    masked_image=images.copy()
    draw=ImageDraw.Draw(masked_image)

    #左側の見えない領域を黒塗り
    draw.rectangle([0, 0, mask_width, height], fill=(0, 0, 0))
    #右側の見えない領域を黒塗り
    draw.rectangle([width - mask_width, 0, width, height], fill=(0,0,0))
    return masked_image

# ==========================================
# 犬の視点
# ==========================================
def dog_vision_filter_flexible(images, is_panorama=False):

    #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=250)
    else:
        wide_img=images

    #画像をnumpy配列にする
    img_array=np.array(wide_img, dtype=float)
    # 2. 画像全体の平均の明るさ（輝度）を計算する
    #    R, G, Bの平均値を出すことで、簡易的に明るさを計算できます
    mean_brightness = np.mean(img_array)
    
    # 基準となる明るさ（この数値を調整して感度を変えられます）
    THRESHOLD = 30.0 

    if mean_brightness<THRESHOLD:
        # ----------------------------------------
        # 🌙 【夜・暗視モード】の処理
        # ----------------------------------------
        # ① 犬の夜の視界に合わせて少し強めにぼかす
        img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=3.7))
        blur_array = np.array(img_blur, dtype=float)
        
        # ② 光を増幅させる（明るさを1.4倍にする）
        # ※ 255を超えないように後で調整します
        night_array = blur_array * 1.4 
        
        # ③ 色を抜き、緑っぽくする（暗視スコープのような表現）
        # モノクロ（輝度）を計算し、それをベースに緑（G）だけを強めにする
        gray = 0.4 * night_array[:,:,0] + 0.7 * night_array[:,:,1] + 0.3 * night_array[:,:,2]
        
        dog_r = gray * 0.94 
        dog_g = gray * 0.94 
        dog_b = gray * 0.94  
     
    else:
        # ==================================================
        # 【ステップ 2】 犬のフィルター処理（ぼかし ＋ 色変換）
        # ==================================================
        # （前後のコード、ぼかし処理などはそのまま）
        # ① 少しだけぼかす
        #img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=4.5))
        #blur_array = np.array(img_blur, dtype=float)
        img_array=np.array(wide_img, dtype=float)

        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]


        #赤を明るさとして還元する処理の強化
        red_luminance=r * 0.25
        #黄色〜緑ベース
        base_yellow_r=r*0.4 + g*0.6 + red_luminance
        base_yellow_g=r*0.3 + g * 0.7 + red_luminance

        #青・水色の厳密判定(「青が明確に強い」または「青と緑が両方高く赤が低い」領域のみ)
        #これにより、白い壁やグレーの物体が誤って青判定されるのを防ぐ
        is_clear_blue=(b>r*1.1)&(b>g*0.8)
        is_cyan=(b>100)&(g>100)&(r<b*0.85)#水色(空や水面)
        is_sky=is_clear_blue |is_cyan

        
        
        #修正２赤以外のベースの色を落とさない(全体が暗くなるのを防止)
        #緑や黄色は明るいまま保ち。赤(Rが高くGやBが低い)だけをグレー(輝度)に置き換える
        is_red=(r > g * 1.25)&(r > b * 1.25)
        #赤い物体の輝度
        red_to_grey=(r*0.3 + g * 0.6 + b * 0.1) 


        #修正３判定に応じてチャンネルごとに色を設定
        #青・水色領域：元のR,Gをしっかり残して青〜水色を綺麗に保つ
        #赤い領域：輝度（）グレーに落とし込んでくすませる
        #それ以外：明るさを保ったまま黄緑ベースにする
        dog_r=np.where(
            is_sky,
            base_yellow_r * 0.25,
            np.where(is_red, red_to_grey * 0.7, base_yellow_r),
        )
        dog_g=np.where(
            is_sky,
            np.maximum(g * 0.8, base_yellow_g * 0.5),
            np.where(is_red, red_to_grey * 0.7, base_yellow_g),
        )

        #青チャンネルは元の鮮やかさと明るさを100％保持
        dog_b=np.where(is_sky, b ,np.minimum(b*1.1,base_yellow_g*0.8 + b*0.3))

        #全体明度を1.2倍に持ち上げ(明るさ補正)
        dog_r=dog_r * 1.2
        dog_g=dog_g * 1.2
        dog_b=dog_b * 1.2


        dog_r=np.clip(dog_r, 0.0, 255.0)
        dog_g=np.clip(dog_g, 0.0, 255.0)
        dog_b=np.clip(dog_b, 0.0, 255.0)


        dog_array=np.stack([dog_r, dog_g, dog_b], axis=-1)
        dog_array=dog_array.astype(np.uint8)

        dog_image=Image.fromarray(dog_array)
        
        #======================-
        #上下ぼかし
        #-======================
        # 【書き換え後】上下でぼかしの強さを変える処理
        # 1. 手前用（強）と奥用（弱）の2つのぼかし画像を作成
        img_blur_strong = dog_image.filter(ImageFilter.GaussianBlur(radius=1.1))  # 手前（下）
        img_blur_weak   = dog_image.filter(ImageFilter.GaussianBlur(radius=10.0))  # 奥（上）

        # 2. 上から下へ変化するグラデーションマスクを作成
        width, height = dog_image.size
        mask = Image.new("L", (width, height))
        draw = ImageDraw.Draw(mask)

        for y in range(height):
            progress=y/height
            alpha = int((progress ** 3) * 255)  # 上:0(黒/弱) 〜 下:255(白/強)
            draw.line([(0, y), (width, y)], fill=alpha)

        # 3. マスクを使って合成し、変数名を合わせる
        dog_image_blur = Image.composite(img_blur_strong, img_blur_weak, mask)

        dog_array=np.array(dog_image_blur)
        dog_r=np.clip(dog_array[:,:,0], 0, 255)
        dog_g=np.clip(dog_array[:,:,1], 0, 255)
        dog_b=np.clip(dog_array[:,:,2], 0, 255)

    # ==================================================
    # 【ステップ 3】 仕上げ
    # ==================================================
    dog_array = np.stack([dog_r, dog_g, dog_b], axis=-1)
    dog_array = np.clip(dog_array, 0, 255).astype(np.uint8)
        
    return Image.fromarray(dog_array)



# ==========================================
# ねこの視点
# ==========================================
def process_cat_vision_flexible(images, is_panorama=False):

    #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=260)
    else:
        wide_img=images
  
    #画像が暗いかどうかを自動判定し、猫の視覚（昼・夜）を切り替える関数
   
    # 1. 画像をNumPyの配列（小数型）にする
    img_array = np.array(wide_img, dtype=float)
    
    # 2. 画像全体の平均の明るさ（輝度）を計算する
    #    R, G, Bの平均値を出すことで、簡易的に明るさを計算できます
    mean_brightness = np.mean(img_array)
    
    # 基準となる明るさ（この数値を調整して感度を変えられます）
    THRESHOLD = 30.0 

        # 3. 【条件分岐】暗い画像かどうかの判定
    if mean_brightness < THRESHOLD:
        # ----------------------------------------
        # 🌙 【夜・暗視モード】の処理
        # ----------------------------------------
        # ① 猫の夜の視界に合わせて少し強めにぼかす
        img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=5))
        blur_array = np.array(img_blur, dtype=float)
        
        # ② 光を増幅させる（一例として明るさを1.8倍にする）
        # ※ 255を超えないように後で調整します
        night_array = blur_array * 1.5 

        # モノクロ（輝度）を計算
        gray = 0.4 * night_array[:,:,0] + 0.7 * night_array[:,:,1] + 0.3 * night_array[:,:,2]
        
        cat_r = gray * 1.31 
        cat_g = gray * 1.35 
        cat_b = gray * 1.35  
        
    else:
        # ----------------------------------------
        # ☀️ 【昼・通常モード】の処理
        # ----------------------------------------
        # ① 少しだけぼかす
        img_array = np.array(wide_img, dtype=float)
        
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]
        
        rgb_max=np.maximum(r, np.maximum(g,b))
        rgb_min=np.minimum(r, np.minimum(g,b))
        is_neutral=(rgb_max - rgb_min) < 30

        #色覚
        red_luminance=r * 0.25
        base_yellow_r=(r * 0.45 + g * 0.55) + red_luminance
        base_yellow_g=(r * 0.35 + g * 0.65) + red_luminance

        #空・水色領域
        is_clear_blue=(b > r * 1.1) & (b > g * 0.8)
        is_cyan=(b > 100) & (g > 100) & (r < b * 0.85)
        is_sky=(is_clear_blue | is_cyan) & (~is_neutral)
        
        #赤い物体の判定
        is_red=(r > g * 1.25) & (r > b * 1.25) & (~is_neutral)
        red_to_gray=r * 0.3 + g * 0.6 + b * 0.1

        cat_r=np.where(
            is_neutral,
            r,
            np.where(
            is_sky,
            base_yellow_r * 0.25,
            np.where(is_red, red_to_gray * 0.7, base_yellow_r),
            ),
        )
        cat_g=np.where(
            is_neutral,
            g,
            np.where(
                is_sky,
                np.maximum(g * 0.8, base_yellow_g * 0.5),
                np.where(is_red, red_to_gray * 0.7, base_yellow_g),
            ),
        )
        cat_b=np.where(
            is_neutral,
            b,
            np.where(
                is_sky, b * 1.15, np.minimum(b * 1.15, base_yellow_g * 0.8 + b * 0.35)
            ),
        )
        #猫特有の補正
        #彩度を押さえて淡いパステル長にする
        cat_r=cat_r * 1.3
        cat_g=cat_g * 1.3
        cat_b=cat_b * 1.3

        cat_array = np.stack([cat_r, cat_g, cat_b], axis=-1)
        cat_array = np.clip(cat_array, 0, 255).astype(np.uint8)

        cat_image=Image.fromarray(cat_array)


        #======================-
        #上下ぼかし
        #-======================
        # 【書き換え後】上下でぼかしの強さを変える処理
        # 1. 手前用（強）と奥用（弱）の2つのぼかし画像を作成
        img_blur_strong = cat_image.filter(ImageFilter.GaussianBlur(radius=1.1))  # 手前（下）
        img_blur_weak   = cat_image.filter(ImageFilter.GaussianBlur(radius=10.0))  # 奥（上）

        # 2. 上から下へ変化するグラデーションマスクを作成
        width, height = cat_image.size
        mask = Image.new("L", (width, height))
        draw = ImageDraw.Draw(mask)

        for y in range(height):
            progress=y/height
            alpha = int((progress ** 3) * 255)  # 上:0(黒/弱) 〜 下:255(白/強)
            draw.line([(0, y), (width, y)], fill=alpha)

        # 3. マスクを使って合成し、変数名を合わせる
        cat_image_blur = Image.composite(img_blur_strong, img_blur_weak, mask)

        cat_array=np.array(cat_image_blur)
        cat_r=np.clip(cat_array[:,:,0])
        cat_g=np.clip(cat_array[:,:,1])
        cat_b=np.clip(cat_array[:,:,2])


    # 4. 仕上げ（数値を0〜255に収めて画像に戻す：前回と同じ共通処理）
    cat_array = np.stack([cat_r, cat_g, cat_b], axis=-1)
    cat_array = np.clip(cat_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(cat_array)



# ==========================================
# うさぎの視点
# ==========================================
def process_rabbit_vision_flexible(images, is_panorama=False):
    #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=360)
    else:
        wide_img=images
    # ----------------------------------------
    # ☀️ 【昼・通常モード】の処理
    # ----------------------------------------
    # ① 少しだけぼかす
    img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=9.3))
    blur_array = np.array(img_blur, dtype=float)
    
    r = blur_array[:, :, 0]
    g = blur_array[:, :, 1]
    b = blur_array[:, :, 2]


    rabbit_r = 0.6 * r + 0.4 * g
    rabbit_g = 0.7 * r + 0.3 * g
    rabbit_b = 0.3 * g + 0.7 * b

    # 4. 仕上げ（数値を0〜255に収めて画像に戻す：前回と同じ共通処理）
    rabbit_array = np.stack([rabbit_r, rabbit_g, rabbit_b], axis=-1)
    rabbit_array = np.clip(rabbit_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(rabbit_array)




#===========================================
#とりの視点
#===========================================
def process_bird_vision_flexible(images, is_panorama=False):
    """
    鳥の視覚(油滴フィルターによる超鮮明な色覚)を再現する関数。
    images:1枚のPIL Image,または複数のPIL Imageが入ったリスト
    """    

            #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=300)
    else:
        wide_img=images

    if not isinstance(wide_img,list):
        images=[wide_img]

    processed_list=[]

    #💡鳥のカラーマトリックス（PillowのRGB順に合わせて配置）
    #自分の色を1.3倍に強調し、隣り合う色を-0.15ずつ引き算して「色の混ざり」をカットする
    bird_matrix=np.array([
        [1.65, -0.2, -0.2],#新しい赤(R)
        [-0.2, 1.65, -0.2],#新しい緑(G)
        [-0.2, -0.2 , 1.65] #新しい青(B)
    ])

    for pil_img in images:
        if pil_img is None:
            continue
        #1.Pillowの画像データを、計算用にNumPy配列(0.0〜1.0)に変換
        img_np=np.array(pil_img).astype(np.float32)/255.0

        #2.OpenCVの高速変換を使ってカラーマトリックス（引き算）を適用
        processed_img=cv2.transform(img_np, bird_matrix)

        #3.計算結果が0未満や１を超えた場合に、0.0〜1.0の範囲に安全に収める
        processed_img=np.clip(processed_img, 0.0, 1.0)

        #4.0〜255の通常の画像データ(整数)に戻す
        bird_vision_np=(processed_img * 255).astype(np.uint8)

        #5.再びPillowのImageオブジェクトに変換してリストに追加
        bird_vision_pil=Image.fromarray(bird_vision_np)

        processed_list.append(bird_vision_pil)

    #💡元々１枚だけだったら単体の画像を返し、複数だったらリストのまま返す
    return processed_list[0] if len(processed_list)==1 else processed_list


#===========================================
#ハムスターの視覚
#===========================================
def process_hamster_vision_flexible(images, is_panorama=False):
    #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=270)
    else:
        wide_img=images
    
    #画像をnumpy配列にする
    img_array=np.array(wide_img, dtype=float)
    # 2. 画像全体の平均の明るさ（輝度）を計算する
    #    R, G, Bの平均値を出すことで、簡易的に明るさを計算できます
    mean_brightness = np.mean(img_array)
    
    # 基準となる明るさ（この数値を調整して感度を変えられます）
    THRESHOLD = 30.0 

    if mean_brightness<THRESHOLD:
        # ----------------------------------------
        # 🌙 【夜・暗視モード】の処理
        # ----------------------------------------
        # ① ハムスターの夜の視界に合わせて少し強めにぼかす
        img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=4.3))
        blur_array = np.array(img_blur, dtype=float)
        
        # ② 光を増幅させる（一例として明るさを1.8倍にする）
        # ※ 255を超えないように後で調整します
        night_array = blur_array * 1.8 
        
        # ③ 色を抜き、緑っぽくする（暗視スコープのような表現）
        # モノクロ（輝度）を計算し、それをベースに緑（G）だけを強めにする
        gray = 0.4 * night_array[:,:,0] + 0.7 * night_array[:,:,1] + 0.3 * night_array[:,:,2]
        
        hamster_r = gray * 0.94  # 赤
        hamster_g = gray * 0.94 # 緑
        hamster_b = gray * 0.94  # 青

    else:
        # ==================================================
        # 【ステップ 2】 ハムスターのフィルター処理（ぼかし ＋ 色変換）
        # ==================================================
        # ① ハムスターの視力に合わせて少しぼかす
        img_blur = wide_img.filter(ImageFilter.GaussianBlur(radius=9.0))
        img_array = np.array(img_blur, dtype=float)
        
        r = img_array[:, :, 0]
        g = img_array[:, :, 1]
        b = img_array[:, :, 2]

        #輝度(人間の近くに近い標準的な明るさを計算)
        gray=0.299 *r + 0.587 * g + 0.114 * b

        # ② ハムスターの色覚計算（赤と緑の区別が苦手な特性）
        hamster_r = 0.85 * gray + 0.15 * r
        hamster_g = 0.85 * gray + 0.10 * g + 0.05 * r
        hamster_b = 0.70 * gray + 0.30 * b

        # ==================================================
        # 【ステップ 3】 仕上げ
        # ==================================================
    hamster_array = np.stack([hamster_r, hamster_g, hamster_b], axis=-1)
    hamster_array = np.clip(hamster_array, 0, 255).astype(np.uint8)
        
    return Image.fromarray(hamster_array)


#===========================================
#魚の視覚
#===========================================
def process_fish_vision_flexible(images, is_panorama=False):
    #画像の＆受け取りとパラノマの視覚処理を入れる
    if is_panorama:
        wide_img=apply_fov_mask(images, fov_degrees=250)
    else:
        wide_img=images
    
    #====================================================--
    #ステップ２　色覚調整＆ぼかし
    #======================================================
    
    #画像をnumpy配列にする
    img_array=np.array(wide_img, dtype=float)

    r=img_array[:,:,0]
    g=img_array[:,:,1]
    b=img_array[:,:,2]
                
    #魚目線の海から見た視線にするための青・緑強調＆低彩度化
    fish_r= r + g * 0.3 + b * 0.3
    fish_g= g + b * 0.3
    fish_b= b + g * 0.3

    fish_array = np.stack([fish_r, fish_g, fish_b], axis=-1)
    fish_array = np.clip(fish_array, 0, 255).astype(np.uint8)

    #2.Numpy配列をPillowのImageオブジェクトに変換する
    fish_img=Image.fromarray(fish_array)
    #ぼかし
    fish_img=fish_img.filter(ImageFilter.GaussianBlur(radius=4.3))

    #================================-
    #ステップ３　魚眼の歪み加工(NumPy一括ベクトル化)
    #================================-
    img_w, img_h=fish_img.size
    #ガラスボールの大きさ
    glassD=400#ガラスボールの直径
    r_val=glassD/2.0    #ガラスボールの半径

    #ガラスボールの加工に使う変数
    cx=img_w/2  #画像中心
    cy=img_h/2
    sx=cx-r_val #左
    sy=cy-r_val #上
    ex=sx+glassD #右
    ey=sy+glassD #下

    #クロップ処理(境界チェック付き)
    area_img=fish_img.crop((int(sx),int(sy),int(ex), int(ey)))
    crop_array=np.array(area_img)

    #グリッド座標の作成(-rからrまで)
    y_coords, x_coords=np.ogrid[-r_val:r_val,-r_val:r_val]

    #中心からの距離lとz値の一括計算
    l_sq=x_coords**2 + y_coords**2
    mask=l_sq<(r_val**2)#円の内側のマスク

    #距離計算(範囲外は安全のため0)
    z=np.zeros_like(l_sq,dtype=np.float32)
    z[mask]=np.sqrt(r_val**2 - l_sq[mask])

    #変換元の参照座標を計算(正しい方向で設定)
    x0=((2 * r_val - z) * x_coords) / (2*r_val) + r_val
    y0=((2 * r_val - z) * y_coords) / (2*r_val) + r_val
    
    #cv2.remap用に型変換(float32)
    map_x=x0.astype(np.float32)
    map_y=y0.astype(np.float32)

    #レンズ外(縁の外側)を黒にするためのマスク適用
    map_x[~mask]=-1
    map_y[~mask]=-1

    #OpenCVの高速マップ再配置関数を実行
    fish_array=cv2.remap(
        crop_array,map_x,map_y,cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0)
    )
    
    return Image.fromarray(fish_array)




#=~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#白内障ボタン
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def apply_cataract_to_animal_vision(animal_pil_img):
    """
    すでに動物の視覚処理（色覚調整・魚眼加工など）が施された画像に対し、
    さらに白内障（水晶体の濁り・くすみ）の効果を重ねがけする関数
    """
    #①輪郭をぼかす(ピンと低下)
    blurred=animal_pil_img.filter(ImageFilter.GaussianBlur(radius=5.0))

    #②コントラストを下げて白く霞ませる(水晶体の白濁)
    enhancer_contrast=ImageEnhance.Contrast(blurred)
    low_contrast=enhancer_contrast.enhance(0.5)

    #③明るさをわずかに上げて濁りを強める
    enhancer_bright=ImageEnhance.Brightness(low_contrast)
    cataract_img=enhancer_bright.enhance(1.1)

    #④ほんのり全体を白・黄色っぽく膜を張る処理(NumPy)
    img_array=np.array(cataract_img,dtype=np.float32)

    #赤と緑のチャンネルを少し持ち上げて、「白濁・黄濁」を表現
    img_array[:,:,0]=np.clip(img_array[:,:,0]*1.05+10,0,255)#Red
    img_array[:,:,1]=np.clip(img_array[:,:,1]*1.03+10,0,255)#Green
    img_array[:,:,2]=np.clip(img_array[:,:,2]+5,0,255)

    return Image.fromarray(img_array.astype(np.uint8))

    

#--- 2. GUIイベントでの処理（切り替え時） ---
# 動物切り替えボタン（「魚」「犬」「猫」など）が押されたタイミングで以下を実行

def update_animal_mode(selected_animal):
    """
    動物が選択されたときに、正常時画像と白内障画像を両方更新する
    """
    global photo_processed,photo_cataract,image_label,original_pil

    #original_pilがまだ読み込まれていない場合の安全装置
    if original_pil is None:
        return
    
    #1.選択された動物の視覚処理を実行
    if selected_animal=="犬":
        current_pil=dog_vision_filter_flexible(original_pil)
    elif selected_animal=="猫":
        current_pil=process_cat_vision_flexible(original_pil)
    elif selected_animal=="うさぎ":
         current_pil=process_rabbit_vision_flexible(original_pil)
    elif selected_animal=="鳥":
         current_pil=process_bird_vision_flexible(original_pil)
    else:
        current_pil=original_pil
    
    cataract_pil=apply_cataract_to_animal_vision(current_pil)


    return current_pil, cataract_pil
    


def process_uploaded_images(file_path, selected_animal):
    # ==================================================
    # 1. 前処理：1枚でも2枚でも、自動的に開いてデータにする
    # ==================================================
    if not uploaded_files:
        return None, None, None
    
    if isinstance(uploaded_files, list):
        if len(uploaded_files)==1:
            img_input=Image.open(uploaded_files[0])
        else:
            img_input=[Image.open(f) for f in uploaded_files]
    else:
        img_input=Image.open(uploaded_files)

    # ==================================================
    # 2. 条件分岐：犬も猫も全く同じように引き渡す
    # ==================================================
    if selected_animal == "犬":
        # 新しく作った自由形の犬フィルターを呼び出す
        processed_img = dog_vision_filter_flexible(img_input, is_panorama=is_panorama)
        
    elif selected_animal == "猫":
        # 自由形の猫フィルターを呼び出す
        processed_img = process_cat_vision_flexible(img_input, is_panorama=is_panorama)
    
    elif selected_animal == "うさぎ":
        processed_img=process_rabbit_vision_flexible(img_input, is_panorama=is_panorama)

    elif selected_animal =="鳥":
        processed_img=process_bird_vision_flexible(img_input, is_panorama=is_panorama)
        
    elif selected_animal=="ハムスター":
        processed_img=process_hamster_vision_flexible(img_input, is_panorama=is_panorama)

    elif selected_animal=="魚":
        processed_img=process_fish_vision_flexible(img_input, is_panorama=is_panorama)

    else:
        # 人間の場合
        processed_img = img_input[0] if isinstance(img_input, list) else img_input
    
    #元の画像(人間視点)
    original_pil=img_input[0] if isinstance(img_input, list) else img_input
    #4.白内障画像の作成(選択された動物の処理画像に対して重ねがけ)
    cataract_img=apply_cataract_to_animal_vision(processed_img)

    return original_pil, processed_img, cataract_img



#============================================
#💡各動物の解説テキストをあらかじめ準備しておく(辞書機能)
#============================================
ANIMAL_EXPLANATIONS={
    "人間":"**【ひとの視覚🔍】**\n**・視力**:1.0~2.0\n**・視野**:180~200°(子供は90°程度)。\n**・色覚**:赤・青・黄色の三色\n・暗視:あり(桿体細胞・暗順応)",
    "犬":"**【いぬの視覚🔍】**\n**・視力**:0.2~0.3(ほぼぼやけている)\n**・視野**:240~270°\n**・色覚**:青と黄色のグラデーション\n　　　(赤や緑はグレーや黄色のように見える)\n**・暗視**:あり\n　　　(タペタム層という、夜でも光を反射して物を見ることが\n　　　できる層がある)\n**・その他**:ある説では犬は紫外線を見ることができるという話や、\n　　　　鳥が持っているとされる磁気センサー(クリプトクローム)\n　　　　があるので、遠くからでも家に帰ってくることができる\n　　　　という話もある。",
    "猫":"**【ねこの視覚🔍】**\n**・視力**:0.1~0.2(ほぼぼやけている)\n**・視野**:200~280°\n**・色覚**:青と緑のグラデーション(赤はグレーのようにみえる。)\n**・暗視**:あり(タペタム層という層があり、星明かり\n　　  　程度の光で鮮明に周囲を見ること\n　　  　ができます。また、薄明視もあります",
    "うさぎ":"**【うさぎの視覚🔍】**\n・**視力**:0.05〜0.1\n**・視野**:355°（鼻の下と真後ろ以外）。\n**・色覚**:青と緑のグラデーション(赤はグレーのように見える。)\n**・暗視**:あり(薄明薄暮性)",
    "鳥":"**【鳥の視覚🔍】** *特にカナリヤ、インコ、文鳥\n**・視力**:2.0~4.0(はやぶさは8.0)\n**・視野**:300°(立体視は苦手)\n**・色覚**:赤・青・黄色+紫外線\n**・暗視**:なし\n**・その他**:クリプトクロームという磁場を感じるセンサー\n　　　　があります。(利用例:伝書鳩での情報伝達)",
    "ハムスター":"**【ハムスターの視覚🔍】**\n**・視力**:0.01\n**・視野**:270°\n**・色覚**:青、緑の二色(赤や黄色はグレーで、視界はほぼ白黒)\n**・暗視**:あり(網膜の97％が光を感じる「桿体細胞」で、薄暗い場所で\n　　　もよく見えます。なので、強い光はストレスの原因になって\nしまいます。",
    "魚":"**【魚の視覚🔍】**\n**・視力**:0.1~0.4°\n**・視野**:320°\n**・色覚**:赤・青・緑(浅い場所に住んでいる魚は一部紫外線も\n　　　見ることができる。)\n・**その他**:水中から水面を通して陸上まで見えており、\n　　　　この現象を「フィッシュウィンドウ」と言います。\n　　　　振動や水流を感知する「側線」という感覚\n　　　　があり、赤・橙・黄色が鈍いと言われています。"
}#\はoptionキー+¥

# ==========================================
# 4. ボタンの表示
# ==========================================

#ボタンが押された時に、選択中の動物を記録して見た目を変える関数。
if "selected_animal" not in st.session_state:
    st.session_state.selected_animal="犬"

st.write("### 動物を選択してください")

#---上段(４列)
col1, col2, col3, col4=st.columns(4)
with col1:
    if st.button("🐶 犬", use_container_width=True):
        st.session_state.selected_animal="犬"
with col2:
    if st.button("🐱 猫", use_container_width=True):
        st.session_state.selected_animal="猫"
with col3:
    if st.button("🐰 うさぎ", use_container_width=True):
        st.session_state.selected_animal="うさぎ"
with col4:
    if st.button("🐤 鳥", use_container_width=True):
        st.session_state.selected_animal="鳥"

#---下段(3列)
col5, col6, col7=st.columns(3)
with col5:
    if st.button("🐹 ハムスター", use_container_width=True):
        st.session_state.selected_animal="ハムスター"
with col6:
    if st.button("🐟 魚", use_container_width=True):
        st.session_state.selected_animal="魚"
with col7:
    if st.button("👤人間", use_container_width=True):
        st.session_state.selected_animal="人間"

#現在選択中の動物を表示・確認
st.caption(f"現在の選択:**{st.session_state.selected_animal}**")


#=========================================================
#画像のドラッグ&ドロップと処理中の表示
#~~=======================================================


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#パーツ配置
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#左右の列(比率3:1)を作成
col_left, col_right=st.columns([3,1])

with col_right:
    st.write("### 操作パネル")

    #ここでトグルスイッチを定義して変数に代入
    show_cataract=st.toggle("白内障の場合")
    show_original=st.toggle("元の画像")

with col_left:
    # 1. ドラッグ＆ドロップ枠
    uploaded_files = st.file_uploader(
        "画像をアップロードしてください",
        type=["jpg", "jpeg", "png","webp","heic"],
        accept_multiple_files=True)
    
    is_panorama=st.checkbox("パノラマ画像ですか？　* **パノラマ画像だと視野範囲も再現できます**")
    
        #関数を通して「１枚にまとめられた加工済み画像」を取得
    selected_animal = st.session_state.selected_animal
    
    if uploaded_files:
        input_images = []
        for f in uploaded_files:
            #関数のreturnに合わせて3つの変数で受け取る
            original_pil, processed_img, cataract_img=process_uploaded_images(uploaded_files, selected_animal)
            if isinstance(f, Image.Image):
                #すでにImage オブジェクトの場合はそのまま使う
                input_images.append(f)
            else:
                input_images.append(Image.open(f))


        # 3. 画像の表示
        if show_cataract:
            st.image(cataract_img, caption="白内障フィルター適用", use_container_width=True)
        elif show_original:
            st.image(input_images[0], caption="元画像（人間の視界）", use_container_width=True)
        else:
            st.image(processed_img, caption=f"{selected_animal}の見え方", use_container_width=True)
        
        # 4. 動物の解説を表示（画像のすぐ下に配置！）
        explanation = ANIMAL_EXPLANATIONS.get(selected_animal, "")
        if explanation:
            #explanation内の\nを\n\nに置換して改行を有効化する
            formatted_explanation=explanation.replace("\n","\n\n")
            st.info(f"{formatted_explanation}")
            
    else:
        st.info("画像をドラッグ＆ドロップするかファイルを選択してください。")

import os
import pandas as pd
import cv2

def build_is_c_k_onlyc_onlyk(error_kind_col_name, crack_names = ['Crack'], knot_names = ['Knot_OK', 'Knot_black', 'Knot_missing'], knot_w_crack=['knot_with_crack']):
    def is_crack(df):
        return True in [c in list(df[error_kind_col_name]) for c in crack_names]
    def is_knot(df):
        return True in [c in list(df[error_kind_col_name]) for c in knot_names]
    def is_knot_with_crack(df):
        return True in [c in list(df[error_kind_col_name]) for c in knot_w_crack]
    def is_only_crack(df):
        return is_crack(df) and not is_knot(df) and not is_knot_with_crack(df)
    def is_only_knot(df):
        return is_knot(df) and not is_crack(df) and not is_knot_with_crack(df)
    return is_crack, is_knot, is_only_crack, is_only_knot

def read_yolo(path):
    return pd.read_csv(path, 
                        sep='\t',
                        index_col=False,
                        names=['kind', '0', '1', '2', '3']
                        )

def show_img_label(label_path, kind):
    img_path = label_path.replace('labels_yololike/', '').replace('_anno.txt', '.bmp')
    _, img_name = os.path.split(img_path)
    print('image: ' +img_path)
    print('annot: ' + path)
    print('label:')
    with open(label_path, 'r') as f:
        print(f.read())
    img = cv2.imread(img_path) 
    img = cv2.resize(img, (500, 300))
    cv2.imshow(kind + img_name, img)
    while True:
        pressed = cv2.waitKey(500)
        if pressed == ord('q'):
            raise
        elif pressed == ord('n'):
            break


if __name__ == '__main__':
    error_col_name = 'kind'
    path = '/home/pedro/datasets/kodytek-woods/labels_yololike'
    show = True

    label_files = [os.path.join(path, x) for x in os.listdir(path)]
    paths_dfs = {path: read_yolo(path) for path in label_files}

    is_crack, is_knot, is_only_crack, is_only_knot = build_is_c_k_onlyc_onlyk('kind')

    only_cracks, only_knots= [], []
    try:
        for path, df in paths_dfs.items():
            flag = False
            if is_only_crack(df):
                only_cracks.append(path)
                flag = True
                if show:
                    show_img_label(path, 'crack ')
            if is_only_knot(df):
                only_knots.append(path)
                if flag:
                    raise Error('is both only crack and only knot! {is_only_crack, is_only_knot} does not wors correctly!')
                # if show:
                #     show_img_label(path, 'knot ')
        print(f'only cracks: {len(only_cracks)} \nonly knots: {len(only_knots)}')
    finally:
        cv2.destroyAllWindows()



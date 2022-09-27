import pygame as pg
import numpy as np
import time
import scipy.io
import os
import math

LIGHT_GREY = (230, 230, 230)

class VisUtils:

    def __init__(self):
        # change this name:
        self.screen_width = 10  # 10
        self.screen_height = 10  # 10
        self.coordinate_scale = 80  # 80
        self.zoom = 1  # 0.25 change the number to adjust the position of the road frame
        self.asset_location = 'assets/'
        self.fps = 24  # max framework

        self.rocket_width = 1.5
        self.rocket_length = 1.5
        self.road_length = 1
        self.coordinate = 'coordinates.png'

        load_path = 'data_rocket_landing.mat'

        self.train_data = scipy.io.loadmat(load_path)

        self.new_data = self.generate(self.train_data)

        self.T = self.new_data['t']

        self.rocket_par = [{'sprite': 'rocket1.png',
                            'state': self.new_data['X'],
                            'orientation': 0}]

        img_width = int(self.rocket_width * self.coordinate_scale * self.zoom)
        img_height = int(self.rocket_length * self.coordinate_scale * self.zoom)

        "initialize pygame"
        pg.init()
        self.screen = pg.display.set_mode((self.screen_width * self.coordinate_scale,
                                           self.screen_height * self.coordinate_scale))

        self.rocket_image = [pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.rocket_par[i]['sprite']),
                                               (img_width, img_height)), -self.rocket_par[i]['orientation']) for i in range(len(self.rocket_par))]

        self.coor_image = pg.image.load(self.asset_location + self.coordinate)

        # we can change the number to adjust the position of the road frame
        self.origin = np.array([0, 4])  # 35, 35; 30, 30;

        # self.origin = np.array([0, 0])

        "Draw Axis Lines"

        self.screen.fill((255, 255, 255))
        # self.draw_axes()  # calling draw axis function
        pg.display.flip()
        pg.display.update()

    def draw_frame(self):
        '''state[t] = [s_x, s_y, v_x, v_y]_t'''
        '''state = [state_t, state_t+1, ...]'''
        # Draw the current frame
        '''frame is counting which solution step'''

        steps = self.T.shape[0]  # 10/0.1 + 1 = 101

        img_width = int(self.rocket_width * self.coordinate_scale * self.zoom)
        img_height = int(self.rocket_length * self.coordinate_scale * self.zoom)

        for k in range(steps - 1):
            self.screen.fill((135, 206, 235))  # grass color: 134, 189, 119; white: 255, 255, 255
            self.draw_axes()

            # Draw Images
            '''getting pos of agent: (x, y)'''
            pos_x_old = np.array(self.rocket_par[0]['state'][0][k])  # car x position
            pos_x_new = np.array(self.rocket_par[0]['state'][0][k + 1])  # get 0 and 1 element (not include 2) : (x, y)

            pos_y_old = np.array(self.rocket_par[0]['state'][2][k])  # car y position
            pos_y_new = np.array(self.rocket_par[0]['state'][2][k + 1])  # get 0 and 1 element (not include 2) : (x, y)
            '''smooth out the movement between each step'''
            pos_x = pos_x_old * (1 - k * 1. / steps) + pos_x_new * (k * 1. / steps)
            pos_y = pos_y_old * (1 - k * 1. / steps) + pos_y_new * (k * 1. / steps)

            pos = (-pos_x, pos_y)  # rocket position

            orientation = -np.array(self.rocket_par[0]['state'][4][k]) * 180 / math.pi

            '''transform pos'''
            pixel_pos_rocket = self.c2p(pos)
            rocket_image = pg.transform.rotate(pg.transform.scale(pg.image.load(self.asset_location + self.rocket_par[0]['sprite']),
                                            (img_width, img_height)), -orientation)

            size_rocket = rocket_image.get_size()

            self.screen.blit(rocket_image, (pixel_pos_rocket[0] - size_rocket[0] / 2, pixel_pos_rocket[1] - size_rocket[1] / 2))
            time.sleep(0.05)

            "drawing the map of state distribution"
            # pg.draw.circle(self.screen, (255, 255, 255), self.c2p(self.origin), 10)  # surface,  color, (x, y),radius>=1

            # time.sleep(1)
            recording_path = 'image_recording/'
            pg.image.save(self.screen, "%simg%03d.png" % (recording_path, k))

            pg.display.flip()
            pg.display.update()

    def draw_axes(self):
        pg.draw.line(self.screen, LIGHT_GREY, self.c2p((1, -0.72)),
                     self.c2p((-1, -0.72)), self.rocket_image[0].get_size()[1]//2)

    def c2p(self, coordinates):
        '''coordinates = x, y position in your environment(vehicle position)'''
        x = self.coordinate_scale * (-coordinates[0] + self.origin[0] + self.screen_width / 2)
        y = self.coordinate_scale * (-coordinates[1] + self.origin[1] + self.screen_height / 2)

        x = int(
            (x - self.screen_width * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_width * self.coordinate_scale * 0.5)
        y = int(
            (y - self.screen_height * self.coordinate_scale * 0.5) * self.zoom
            + self.screen_height * self.coordinate_scale * 0.5)
        '''returns x, y for the pygame window'''

        return np.array([x, y])

    def generate(self, data):
        t_bar = np.linspace(0, 3, 50)
        X_bar = np.zeros((5, t_bar.shape[0]))
        i = 0
        j = 0
        time = 0
        t = data['t']  # time is from train_data
        X = data['X']
        while time <= 3.0:
            while t[0][i] <= time:
                i += 1
            """
            rocket: original state: (px, vx, py, vy, theta)
            scaling the position and velocity to have a better animation, original position and velocity is in [0, 1]
            """
            X_bar[0][j] = ((time - t[0][i - 1]) * (X[0][i] - X[0][i - 1]) / (t[0][i] - t[0][i - 1]) + X[0][i - 1]) * 6
            X_bar[1][j] = ((time - t[0][i - 1]) * (X[1][i] - X[1][i - 1]) / (t[0][i] - t[0][i - 1]) + X[1][i - 1]) * 6
            X_bar[2][j] = ((time - t[0][i - 1]) * (X[2][i] - X[2][i - 1]) / (t[0][i] - t[0][i - 1]) + X[2][i - 1]) * 6
            X_bar[3][j] = ((time - t[0][i - 1]) * (X[3][i] - X[3][i - 1]) / (t[0][i] - t[0][i - 1]) + X[3][i - 1]) * 6
            X_bar[4][j] = (time - t[0][i - 1]) * (X[4][i] - X[4][i - 1]) / (t[0][i] - t[0][i - 1]) + X[4][i - 1]
            time = time + 0.1
            j += 1

        new_data = dict()
        new_data.update({'t': t_bar,
                         'X': X_bar})

        return new_data

if __name__ == '__main__':
    vis = VisUtils()
    vis.draw_frame()

    path = 'image_recording/'
    import glob

    image = glob.glob(path + "*.png")
    img_list = image  # [path + "img" + str(i).zfill(3) + ".png" for i in range(episode_step_count)]

    import imageio

    images = []
    for filename in img_list:
        images.append(imageio.imread(filename))
    tag = 'rocket_landing'
    imageio.mimsave(path + 'movie_' + tag + '.gif', images, 'GIF', fps=10)
    # Delete images
    [os.remove(path + file) for file in os.listdir(path) if ".png" in file]

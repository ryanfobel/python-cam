import gtk
import sys
import os
import argparse

from path import path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import find
import cv2
import cv

# states
SET_BASELINE = 0
SET_ROI = 1

class DropWindow(object):
    def __init__(self, images, output):
        self.output_file = output
        self.state = SET_BASELINE
        self.contact_line = []
        self.baseline = None
        self.roi_anchor = []
        self.contact_angles = []
        self.background = None
        if type(images)==type(cv2.VideoCapture()):
            self.n_frames = int(images.get(cv.CV_CAP_PROP_FRAME_COUNT))
            print "Video has %d frames." % self.n_frames
            f, image = images.read()
        else:
            self.n_frames = 1
            image = images
        self.frame = 1
        h, w = image.shape[0:2]
        self.roi = [0, 0, w, h]
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_title('Contact Angle Measurement')
        self.window.set_resizable(False)
        self.window.connect("delete_event", self.on_delete_event)
        self.window.connect("destroy", self.on_destroy)
        self.image = gtk.Image()
        self.draw()
        self.image.show()
        self.window.add(self.image)
        self.window.connect("button_press_event", self.on_button_press_event)
        self.window.connect("button_release_event", self.on_button_release_event)
        self.window.connect("motion_notify_event", self.on_motion_notify_event)
        self.window.connect("key_press_event", self.on_key_press_event)
        self.window.set_events(gtk.gdk.BUTTON_PRESS_MASK |
                               gtk.gdk.BUTTON_RELEASE_MASK |
                               gtk.gdk.BUTTON_MOTION_MASK |
                               gtk.gdk.KEY_PRESS_MASK)
        self.window.show()
        self.set_background(image)

    def set_background(self, image):
        gray = cv2.cvtColor(image, cv.CV_BGR2GRAY)
        self.edges = cv2.Canny(gray, 100, 200)
        channel = image[:,:,1]
        channel[self.edges>0] = 0
        image[:,:,1] = channel
        image[:,:,2] = channel
        channel[self.edges>0] = 255
        image[:,:,0] = channel
        self.background = image
        self.recalculate()
        self.draw()

    def on_delete_event(self, widget, event, data=None):
        return False

    def on_destroy(self, widget, data=None):
        self.frame = self.n_frames
        gtk.main_quit()

    def on_button_press_event(self, widget, event):
        p = (int(event.x), int(event.y))
        if self.state==SET_BASELINE:
            if len(self.contact_line)<2:
                self.contact_line.append(p)
            else:
                # replace the point that is closest
                d = np.sqrt(np.sum((self.contact_line - np.vstack((p,p)))**2, 1))
                self.contact_line[find(np.min(d)==d)[0]]=p
                self.recalculate()
        elif self.state==SET_ROI:
            self.roi_anchor = p
            self.roi = np.array([p[0], p[1], 0, 0])
        self.draw()

    def on_button_release_event(self, widget, event):
        if self.state==SET_ROI:
            self.recalculate()
            self.draw()

    def on_motion_notify_event(self, widget, event):
        x, y = (int(event.x), int(event.y))
        state = event.state
        if state & gtk.gdk.BUTTON1_MASK and self.state==SET_ROI:
            self.roi[2] = abs(self.roi_anchor[0]-x)
            if x<=self.roi_anchor[0]:
                self.roi[0] = x
            else:
                self.roi[0] = self.roi_anchor[0]
            self.roi[3] = abs(self.roi_anchor[1]-y)
            if y<=self.roi_anchor[1]:
                self.roi[1] = y
            else:
                self.roi[1] = self.roi_anchor[1]
            self.draw()

    def on_key_press_event(self, widget, event):
        keyname = gtk.gdk.keyval_name(event.keyval)
        
        def print_results():
            results = self.contact_angle_string()
            print results
            if self.output_file:
                with open(self.output_file, 'a') as f:
                    f.write(results+"\n")

        if keyname=='Return':
            print_results()
            while self.frame < self.n_frames:
                f, image = images.read()
                self.set_background(image)
                self.frame += 1
                print_results()
                while gtk.events_pending():
                    gtk.main_iteration()
        elif keyname=='r':
            self.state = SET_ROI
        elif keyname=='b':
            self.state = SET_BASELINE

    def in_roi(self, x, y):
        if x>self.roi[0] and x<self.roi[0]+self.roi[2] and \
            y>self.roi[1] and y<self.roi[1]+self.roi[3]:
            return True
        return False

    def contact_angle_string(self):
        return '%d, ' % (self.frame) + \
            ', '.join(['%.1f' % (a*180.0/np.pi) for a in self.contact_angles])

    def recalculate(self):
        self.baseline = None

    def draw(self):
        if self.background is None:
            return
        pixbuf = gtk.gdk.pixbuf_new_from_array(self.background,
                                               gtk.gdk.COLORSPACE_RGB, 8)
        pixmap, mask = pixbuf.render_pixmap_and_mask()
        cm = pixmap.get_colormap()
        white = cm.alloc_color('white')
        red = cm.alloc_color('red')
        green = cm.alloc_color('green')

        if len(self.roi):
            gc = pixmap.new_gc(foreground=white)
            x, y, w, h = self.roi
            pixmap.draw_rectangle(gc, False, x, y, w, h)

        if len(self.contact_line)==2:
            x0, y0 = self.contact_line[0]
            x1, y1 = self.contact_line[1]
            w = pixbuf.get_width()

            if self.baseline is None:
                slope = float(y1-y0)/float(x1-x0)
                y_int = y0-slope*x0
                self.baseline = np.poly1d([slope, y_int])
                #print "baseline: y = %.3fx+%.3f" % (slope, y_int)

                # unit vector perpendicular to the line
                v = np.array((-y1+y0, x1-x0)) / \
                    np.sqrt((x1-x0)**2 + (y1-y0)**2)

                # set the direction of the vector to point up (-y)
                if v[1]>0:
                    v *= -1
                #print "unit vector perpendicular to baseline: v =", v

                edge_points = find(self.edges)
                points = []
                for ind in edge_points:
                    (y, x) = np.unravel_index(ind, self.background.shape[0:2])
    
                    # calculate the distance of each point to the line (+ve distances are
                    # above the line)
                    d = np.dot(np.array((x-x0, y-y0)), v)
        
                    # only keep points which are above the line and within the roi
                    if d>0 and self.in_roi(x, y):
                        points.append((ind, x, y, d))
                points = np.array(points)

                paths = []
                self.tangents = []
                self.base_points = []
                contact_angles = []

                # calculate the distance between each of the points
                n_points = points.shape[0]
                if n_points>0:
                    Y = np.tile(points[:,2], (n_points, 1))
                    X = np.tile(points[:,1], (n_points, 1))
                    D = np.sqrt((X-np.transpose(X))**2 + (Y-np.transpose(Y))**2)

                    checked = np.zeros(n_points, dtype=bool)

                    # group connected pixels together
                    def find_connected(i):
                        if checked[i]:
                            return
                        paths[-1].append(points[i])
                        checked[i] = True
                        for ind in find(D[i]<=np.sqrt(2)):
                            find_connected(ind)

                   # create paths of connected points that intersect with the baseline
                    for i in find(points[:,3]<=np.sqrt(2)):
                        if checked[i]:
                            continue
                        # create a new path
                        paths.append([])
                        find_connected(i)

                    for i in range(len(paths)):
                        paths[i] = np.array(paths[i])

                        # fit a polynomial to the points in each path
                        p_fit = np.poly1d(np.polyfit(paths[i][:,2], paths[i][:,1], 3))

                        # calculate the slope through the base point
                        ind = find(min(paths[i][:, 3])==paths[i][:, 3])[0]
                        y_base = int(round(paths[i][ind, 2]))
                        x_base = int(round(p_fit(y_base)))
                        self.base_points.append((x_base, y_base))
                        slope = 1.0/p_fit.deriv(1)(y_base)
                        y_int = y_base-slope*x_base
                        tangent = np.poly1d([slope, y_int])
                        self.tangents.append(tangent)

                        # calculate the contact angle
                        a = np.array([self.baseline(1)-self.baseline(0), 1])
                        b = np.array([tangent(1)-tangent(0), 1])
                        contact_angle = np.arccos(np.dot(a, b)/ \
                            (np.linalg.norm(a)*np.linalg.norm(b)))
                        if (x_base<self.background.shape[1]/2 and b[0]>0) or \
                           (x_base>self.background.shape[1]/2 and b[0]<0):
                            contact_angle = np.pi-contact_angle
                        contact_angles.append(contact_angle)

                self.paths = paths
                self.contact_angles = np.array(contact_angles)
                print self.contact_angle_string()

            # draw the baseline
            gc = pixmap.new_gc(foreground=red)
            if self.baseline.order==1:
                y_int = int(round(self.baseline.coeffs[1]))
            else:
                y_int = int(round(self.baseline.coeffs[0]))
            pixmap.draw_line(gc, 0, y_int, w, int(round(self.baseline(w))))
            pixmap.draw_rectangle(gc, True, x0-2, y0-2, 4, 4)
            pixmap.draw_rectangle(gc, True, x1-2, y1-2, 4, 4)

            for i, path in enumerate(self.paths):
                # draw the points used for the fit
                gc = pixmap.new_gc(foreground=red)

                # draw the fitted polynomial
                p_fit = np.poly1d(np.polyfit(self.paths[i][:,2],
                    self.paths[i][:,1], 3))
                for y in path[:,2]:
                    x = int(round(p_fit(y)))
                    y = int(round(y))
                    pixmap.draw_line(gc, x, y, x, y)

                # draw the tangents
                gc = pixmap.new_gc(foreground=green)
                x_base, y_base = self.base_points[i]
                y = self.roi[1]
                if self.tangents[i].order==1:
                    slope, y_int = self.tangents[i].coeffs
                else:
                    slope = 0
                    y_int = self.tangents[i].coeffs[0]
                x = int(round((y-y_int)/slope))
                pixmap.draw_line(gc, x_base, y_base, x, y)

            gc = pixmap.new_gc(foreground=green)
        self.image.set_from_pixmap(pixmap, mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze contact angles from a video or image.')
    parser.add_argument('-i', '--input', help='input video or image file to process')
    parser.add_argument('-o', '--output', help='output csv file')
    args = parser.parse_args()

    if args.input is None:
        filename = 'sample.jpg'
    else:
        filename = args.input

    ext = filename.split('.')[-1]
    if ext=='jpg' or ext=='png':
        images = cv2.imread(filename)
    elif ext=='avi':
        images = cv2.VideoCapture(filename)

    window = DropWindow(images, args.output)
    gtk.main()

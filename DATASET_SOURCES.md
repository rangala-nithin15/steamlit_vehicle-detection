# Dataset sources and labeling plan

## Pexels parking clips (review pool)

Pexels license: https://www.pexels.com/license/ (free use; attribution not required; identifiable people must not be shown in a bad light). Source search: https://www.pexels.com/search/videos/car%20parking/ . These are not theft-labeled clips. Download a clip from its Pexels page and place it under data/review/pexels/. Review the entire clip before moving it into data/videos/normal/. Do not label anyone in a Pexels clip as a thief. Some results are aerial parking-lot views and are useful only as scene/background examples, not owner/person interaction examples.

Suggested source pages (20 clips; titles/pages are supplied so downloads can be chosen and credited):

1. https://www.pexels.com/video/aerial-footage-of-a-full-parking-lot-5982899/
2. https://www.pexels.com/video/car-parking-beach-27703164/
3. https://www.pexels.com/video/an-open-space-parking-lot-2406463/
4. https://www.pexels.com/video/a-car-parking-outside-a-funfair-8470014/
5. https://www.pexels.com/video/car-parking-at-night-11469311/
6. https://www.pexels.com/video/automated-vertical-car-parking-system-29927092/
7. https://www.pexels.com/video/drone-footage-of-car-parking-11496453/
8. https://www.pexels.com/video/aerial-view-of-industrial-park-and-parking-lot-35035721/
9. https://www.pexels.com/video/aerial-view-of-large-black-and-white-car-parking-lot-33763794/
10. https://www.pexels.com/video/aerial-view-of-a-parking-lot-with-cars-18828536/
11. https://www.pexels.com/video/parking-spaces-in-a-commercial-area-5607783/
12. https://www.pexels.com/video/drone-footage-of-cars-parked-and-driving-on-parking-lot-of-the-mall-5607777/
13. https://www.pexels.com/video/a-parking-lot-with-cars-parked-in-it-27668854/
14. https://www.pexels.com/video/a-parking-lot-with-cars-parked-in-it-19862862/
15. https://www.pexels.com/video/a-man-is-driving-a-car-in-the-parking-lot-27668915/
16. https://www.pexels.com/video/close-up-of-parked-car-on-empty-road-29158610/
17. https://www.pexels.com/video/people-walking-in-the-parking-lot-4609915/
18. https://www.pexels.com/video/people-walking-around-the-carpark-4143289/
19. https://www.pexels.com/video/aerial-view-of-suburban-street-with-parked-car-33700597/
20. https://www.pexels.com/video/an-aerial-view-of-a-parking-lot-with-cars-19201622/

## EPFL Mini-drone dataset (research theft examples)

Official page and download instructions: https://www.epfl.ch/labs/mmspg/downloads/mini-drone/ . It contains 38 annotated parking-lot scenarios (16–24 seconds) grouped as normal, suspicious, and illicit. The project page says research use is permitted with acknowledgement of the specified 2015 paper; it excludes commercial distribution/use. Follow the official page's current access instructions and keep its attribution with the project.

After obtaining the videos and VideoDataset_Description.xls, inspect the sequence descriptions/annotations. Copy only clear ordinary parking/arrival clips into data/videos/normal/. Copy a clip into data/videos/detected/ only if its annotation explicitly describes stealing a vehicle/car. Exclude suspicious-only behavior, theft of objects from a car, fighting, and mis-parking from this vehicle-theft binary task. Keep a source/sequence manifest so related clips are never split across train/validation/test. The current trainer splits individual files, so do not put multiple cuts from one scene in the same run until grouped splitting is added.

## Train

The trainer currently expects labeled files in:

- data/videos/normal/
- data/videos/detected/

It requires at least 10 clips in each class. Do not train until each clip has been reviewed and assigned the correct label. Pexels alone cannot provide a detected/theft class. Treat resulting metrics as a small prototype evaluation, not real-world performance.

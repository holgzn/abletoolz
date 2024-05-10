"""Ableton set parsing."""
import copy
import enum
import functools
import gzip
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
from typing import Callable, Dict, Generator, List, Optional, ParamSpec, Tuple, TypeVar
from xml.etree import ElementTree as ET

from abletoolz import color_tools, utils
from abletoolz.ableton_track import AbletonTrack
from abletoolz.misc import CB, RB, RST, STEREO_OUTPUTS, B, C, G, M, R, Y, get_element, ElementNotFound, find_parent

if sys.platform == "win32":
    import win32_setctime

if sys.platform == "darwin":
    import plistlib

logger = logging.getLogger(__name__)


class SetError(Exception):
    """Ableton set parse error."""


class SetOperatingSystem(enum.Enum):
    """Pre ableton 11, sets store data differently.

    Sets do not store any OS information, but we can guess based on encoding of data, AU units vs VSTs and some
    other differences.
    """

    MAC_OS = enum.auto()
    WINDOWS_OS = enum.auto()
    UNSET = enum.auto()


def version_supported(set_version: Tuple[int, int, int], supported_version: Tuple[int, int, int]) -> bool:
    """Check if set version is supported for method."""
    for set_v, supported_v in zip(set_version, supported_version):
        if set_v > supported_v:
            return True
        elif set_v < supported_v:
            return False
    return True


P = ParamSpec("P")
RT = TypeVar("RT")


def above_version(supported_version: Tuple[int, int, int]) -> Callable[[Callable[P, RT]], Callable[P, RT]]:
    """Decorator factory to handle method support for changing XML schemas across Ableton versions.

    https://help.ableton.com/hc/en-us/articles/360000841004-Backward-Compatibility
    """

    def wrapper(f: Callable[P, RT]) -> Callable[P, RT]:
        @functools.wraps(f)
        def wrapped_func(*args: P.args, **kwargs: P.kwargs) -> RT:
            # arg[0] is self
            if not version_supported(args[0].version_tuple, supported_version):  # type: ignore[attr-defined]
                logger.info("Function %s is only supported for %s and above.", f.__name__, supported_version)
                raise SetError(f"{f.__name__} not supported for this version!")
            return f(*args, **kwargs)

        return wrapped_func

    return wrapper


def set_loaded(f: Callable[..., RT]) -> Callable[..., RT]:
    """Decorator that checks set XML has been loaded into Xtree."""

    @functools.wraps(f)
    def wrapped_func(self: "AbletonSet", *args, **kwargs) -> RT:
        if self.root is None:
            raise SetError("Set is not loaded!")
        return f(self, *args, **kwargs)

    return wrapped_func


def elements_equal(e1: ET.Element, e2: ET.Element) -> bool:
    """Check if two xml.Etree roots are equivalent."""
    if e1.tag != e2.tag:
        return False
    if e1.text != e2.text:
        return False
    if e1.tail != e2.tail:
        return False
    if e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


class AbletonSet(object):
    """Set object."""

    def __init__(self, pathlib_obj: pathlib.Path) -> None:
        """Construct class."""
        self.name = pathlib_obj.name
        self.path = pathlib_obj
        self.tree = None
        self.root: Optional[ET.Element] = None

        # Parsed set variables.
        self.project_root_folder: Optional[pathlib.Path] = None  # Folder where Ableton Project Info resides.
        self.last_modification_time: Optional[float] = None
        self.creation_time: Optional[float] = None

        self.version: Optional[str] = None  # Official Ableton live version.
        self.version_tuple: Optional[Tuple[int, int, int]] = None

        self.tempo: Optional[float] = None
        self.furthest_bar: Optional[int] = None
        self.bpm: Optional[float] = None
        self.tracks: List[AbletonTrack] = []
        self.sample_list: List[utils.SampleRef] = []

        self.mainId = "Main"

        # TODO WIP, not finished/used necessarily.
        self.set_os: SetOperatingSystem = SetOperatingSystem.UNSET
        self.missing_absolute_samples: List[pathlib.Path] = []
        self.missing_relative_samples: List[pathlib.Path] = []
        self.found_vst_dirs: List[pathlib.Path] = []
        self.last_elem = None
        self.key = None

    def __eq__(self, o: object) -> bool:
        """Compare two sets."""
        if self.root is None:
            return False
        if not isinstance(o, AbletonSet) or isinstance(o, AbletonSet) and o.root is None:
            return False
        return elements_equal(self.root, o.root)

    def open_folder(self) -> None:
        """Open folder in file explorer/finder.

        Currently unused.
        """
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select, "{self.path}"')
        elif sys.platform == "darwin":
            subprocess.Popen(f"open {self.path}")

    @set_loaded
    def load_version(self) -> None:
        """Load version."""
        assert self.root  # Shut mypy up since decorator checks this.
        self.version = self.root.get("Creator")
        if not isinstance(self.version, str):
            raise SetError("Couldn't parse Creator from set.")
        parsed = re.findall(
            r"Ableton Live ([0-9]{1,2})\.([0-9]{1,3})[\.b]{0,1}([0-9]{1,3}){0,1}",
            self.version,
        )
        if not parsed:
            raise SetError("Couldn't parse set version!")
        parsed = [int(x) if x.isdigit() else x for x in parsed[0] if x != ""]
        if len(parsed) == 3:
            major, minor, patch = parsed
        elif len(parsed) == 2:
            major, minor = parsed
            patch = 0
        else:
            raise SetError(f"Could not parse version from: {self.version}")
        self.version_tuple = major, minor, patch
        logger.info("%sSet version: %s%s", B, M, self.version)
        if major < 12:
            self.mainId = "Master"
        if "b" in self.version.split()[-1]:
            logger.warning("%sSet is from a beta version, some commands might not work properly!", Y)

    def parse(self) -> bool:
        """Uncompresses ableton set and loads into element tree."""
        with open(self.path, "rb") as fd:
            first_two_bytes = fd.read(2)
            if first_two_bytes == b"\xab\x1e":  # yes, it spells able :P
                logger.error("%s%sIs pre Ableton 8.2.x which is unsupported.", R, self.path)
                return False
            elif first_two_bytes != b"\x1f\x8b":
                logger.error(
                    "%s%sFile is not .als or is an older format that doesn't use gzip!, cannot open...", R, self.path
                )
                return False
        self.get_file_times()
        with gzip.open(self.path, "r") as fd:
            data = fd.read().decode("utf-8")
            if not data:
                logger.error("%sError loading data %s!", R, self.path)
                return False
            self.root = ET.fromstring(data)
            return True

    def find_project_root_folder(self) -> Optional[pathlib.Path]:
        """Find project root folder for set."""
        # TODO Parse project .cfg file and logger.info information.
        if self.project_root_folder:
            return self.project_root_folder

        max_folder_search_depth = 10
        for i, current_dir in enumerate(self.path.parents):
            if i > max_folder_search_depth:
                logger.warning("%sReached maximum search depth, exiting..", R)
                break
            elif pathlib.Path(current_dir / "Ableton Project Info").exists():
                self.project_root_folder = current_dir
                logger.debug("%sProject root folder: %s", C, current_dir)
                return self.project_root_folder
        logger.error("%sCould not find project folder(Ableton Project Info), unable to validate relative paths!", R)
        return None

    def generate_xml(self) -> bytes:
        """Add header and footer to xml data."""
        if self.root is None:
            raise SetError("Set is not loaded!")
        header = '<?xml version="1.0" encoding="UTF-8"?>\n'.encode("utf-8")
        footer = "\n".encode("utf-8")
        xml_output = ET.tostring(self.root, encoding="utf-8")
        return header + xml_output + footer

    def save_xml(self) -> None:
        """Save set XML."""
        xml_file = self.path.parent / (self.path.stem + ".xml")
        if xml_file.exists():
            utils.create_backup(xml_file)
        with xml_file as fd:
            fd.write_bytes(self.generate_xml())
        logger.info("%sSaved xml to %s", G, xml_file)

    def get_file_times(self) -> None:
        """Find set creation/modification times."""
        if sys.platform == "win32" or sys.platform == "linux":
            self.creation_time = os.path.getctime(self.path)
        else:
            self.creation_time = os.stat(self.path).st_birthtime
        self.last_modification_time = os.path.getmtime(self.path)
        logger.debug(
            "%sFile creation time %s, Last modification time: %s",
            B,
            utils.format_date(self.creation_time),
            utils.format_date(self.last_modification_time),
        )

    def restore_file_times(self, pathlib_obj: pathlib.Path) -> None:
        """Restore original creation and modification times to file."""
        if self.last_modification_time is None:
            logger.warning("No modification time! Can't restore original time...")
            return
        os.utime(self.path, (self.last_modification_time, self.last_modification_time))
        if sys.platform == "win32":
            win32_setctime.setctime(self.path, self.creation_time)
        elif sys.platform == "darwin":
            date = utils.format_date(self.creation_time)
            path = str(pathlib_obj).replace(" ", r"\ ")
            os.system(f'SetFile -d "{date}" {path} >/dev/null')
        logger.debug(
            "%sRestored set creation and modification times: %s, %s",
            G,
            utils.format_date(self.creation_time),
            utils.format_date(self.last_modification_time),
        )

    def write_set(self) -> None:
        """Recompresses set to gzip. Used in thread to help prevent file getting corrupted mid write."""
        with gzip.open(self.path, "wb") as fd:
            fd.write(self.generate_xml())
        logger.info("%sSaved set to %s", G, self.path)
        self.restore_file_times(self.path)

    def save_set(self, append_bars_bpm: bool = False, prepend_version: bool = False, append_name: str = None, new: bool = False) -> None:
        """Save set to disk with optional filename modifications.

        This function saves the current set to disk, first creating a backup of the original file.
        It optionally appends the number of bars and BPM to the filename, and/or prepends the version number.
        The actual writing of the set to disk is performed in a separate thread.

        Args:
            append_bars_bpm: If True, append the number of bars and BPM to the filename.
            prepend_version: If True, prepend the version number to the filename.
        """
        if new:
            if not (append_name or append_bars_bpm or prepend_version):
                raise ValueError("Cannot create a new file wihout adding something to the name")
        if not new:
            utils.create_backup(self.path)
        if append_bars_bpm:
            cleaned_name = re.sub(r"_\d{1,3}bars_\d{1,3}\.\d{2}bpm", "", self.path.stem)
            new_filename = cleaned_name + f"_{self.furthest_bar}bars_{self.bpm:.2f}bpm.als"
            self.path = pathlib.Path(self.path.parent / new_filename)
            logger.debug("%sAppending bars and bpm, new set name: %s.als", M, self.path.stem)

        if append_name is not None:
            cleaned_name = re.sub(r"\.als$", "", self.path.stem)
            to_append = re.sub(r"\s", "_", append_name)
            new_filename =  f"{cleaned_name}_{to_append}.als"
            self.path = pathlib.Path(self.path.parent / new_filename)
            logger.debug("%sAppending custom text, new name %s.als", M, self.path.stem)


        if self.version_tuple and prepend_version:
            version_string = f"{self.version_tuple[0]}.{self.version_tuple[1]}.{self.version_tuple[2]}_"
            cleaned_name = re.sub(r"\d{1,2}\.\d{1,3}\.[b\d]{1,5}_", "", self.path.stem)
            self.path = self.path.parent / (version_string + cleaned_name + self.path.suffix)

        # Create non daemon thread so that it is not forcibly killed if parent process is killed.
        thread = threading.Thread(target=self.write_set)
        thread.start()
        thread.join()

    # Data parsing functions.
    @above_version(supported_version=(8, 0, 0))
    def load_tracks(self) -> None:
        """Load tracks into AbletonTrack src."""
        if len(self.tracks) == 0:
            tracks = get_element(self.root, "LiveSet.Tracks")
            for track in tracks:
                self.tracks.append(AbletonTrack(track, self.version_tuple))
            self.tracks.append(AbletonTrack(get_element(self.root, f"LiveSet.{self.mainId}Track"), self.version_tuple))

    def print_tracks(self, isExportMode=False) -> None:
        """logger.infos track info."""
        if not isExportMode:
            logger.info("Tracks:\n%s", "\n".join([str(x) for x in self.tracks]))
        else:
            num = 0
            group_stack = []
            indent = ""
            for track in self.tracks:
                num = num + 1
                display_name = track.name
                if display_name.startswith("# "):
                    display_name = display_name.replace("# ",str(num)+" ")
                if len(group_stack) > 0 and (track.group_id != group_stack[-1] or track.group_id == "-1"):
                    group_stack.pop()
                    indent = indent[0:-2]
                elif track.group_id != "-1" and (len(group_stack) == 0  or  track.group_id != group_stack[-1]):
                    group_stack.append(track.group_id)
                    indent = indent+"  "
                print(indent + display_name)



    @set_loaded
    def find_furthest_bar(self) -> int:
        """Find the max of the longest clip or furthest bar something is in Arrangement."""
        assert self.root is not None  # Shut mypy up.
        current_end_times = [int(float(end_times.get("Value", 0))) for end_times in self.root.iter("CurrentEnd")]
        self.furthest_bar = int(max(current_end_times) / 4) if current_end_times else 0
        return self.furthest_bar

    @above_version(supported_version=(8, 2, 0))
    def get_bpm(self) -> float:
        """Get bpm from Ableton Live set XML."""
        if self.version_tuple is None:
            raise SetError("Set version is not parsed!")
        post_10_bpm = f"LiveSet.{self.mainId}Track.DeviceChain.Mixer.Tempo.Manual"
        pre_10_bpm = "LiveSet.MasterTrack.MasterChain.Mixer.Tempo.ArrangerAutomation.Events.FloatEvent"
        pre_10_bpm = "LiveSet.MasterTrack.DeviceChain.Mixer.Tempo.ArrangerAutomation.Events.FloatEvent"
        major, minor, _ = self.version_tuple
        if major >= 10 or major >= 9 and minor >= 7:
            bpm_elem = get_element(self.root, post_10_bpm, attribute="Value", silent_error=True)
        else:
            bpm_elem = get_element(self.root, pre_10_bpm, attribute="Value")
        self.bpm = round(float(bpm_elem), 6)
        return self.bpm

    def estimate_length(self) -> None:
        """Multiply the longest bar with length per bar by inverting BPM."""
        if self.bpm is None or self.furthest_bar is None:
            logger.info("%sCan't estimate length without bpm and furthest bar.", R)
            return
        # TODO improve this to find the time signature from the set and use it here instead of only 4/4.
        seconds_total = ((4 * int(self.furthest_bar)) / self.bpm) * 60
        length = f"{int(seconds_total // 60)}:{round(seconds_total % 60):02d}"
        logger.info(
            "%sLongest clip or furthest arrangement position: %s bars. %sEstimated length(Only valid for 4/4): %s",
            M,
            self.furthest_bar,
            C,
            length,
        )

    @set_loaded
    def set_track_heights(self, height: int) -> None:
        """In Arrangement view, sets all track lanes/automation lanes to specified height."""
        assert self.root is not None  # Shutup mypy, not possible at runtime.
        height = min(425, (max(17, height)))  # Clamp to valid range.
        for el in self.root.iter("LaneHeight"):
            el.set("Value", str(height))
        logger.info("%sSet track heights to %s.", G, height)

    @set_loaded
    def set_track_widths(self, width: int) -> None:
        """Set all track widths in Clip view to specified width."""
        assert self.root is not None  # Shutup mypy.
        width = min(264, (max(17, width)))  # Clamp to valid range.
        # Sesstion is how it's named in the set, not a typo!
        for el in self.root.iter("ViewStateSesstionTrackWidth"):
            el.set("Value", str(width))
        logger.info("%sSet track widths to %s.", G, width)

    @set_loaded
    def fold_tracks(self) -> None:
        """Fold all tracks."""
        assert self.root is not None  # Shutup mypy.
        for el in self.root.iter("TrackUnfolded"):
            el.set("Value", "false")
        logger.info("%sFolded all tracks.", G)

    @set_loaded
    def unfold_tracks(self) -> None:
        """Unfold all tracks."""
        assert self.root is not None  # Shutup mypy, not possible at runtime.
        for el in self.root.iter("TrackUnfolded"):
            el.set("Value", "true")
        logger.info("%sUnfolded all tracks.", G)

    @above_version(supported_version=(8, 2, 0))
    def set_audio_output(self, output_number: int, element_string: str) -> None:
        """Set audio output."""
        if output_number not in STEREO_OUTPUTS:
            raise ValueError(f"{R}Output number invalid!. Available options: \n{STEREO_OUTPUTS}{RST}")
        output_obj = STEREO_OUTPUTS[output_number]
        out_target_element = get_element(
            self.root,
            f"LiveSet.{element_string}.DeviceChain.AudioOutputRouting.Target",
            silent_error=True,
        )
        if not isinstance(out_target_element, ET.Element):
            out_target_element = get_element(  # ableton 8 sets use "MasterChain" for master track.
                self.root,
                f"LiveSet.{element_string}.{self.mainId}Chain.AudioOutputRouting.Target",
            )
            lower_display_string_element = get_element(
                self.root,
                f"LiveSet.{element_string}.{self.mainId}Chain.AudioOutputRouting.LowerDisplayString",
            )
        else:
            lower_display_string_element = get_element(
                self.root,
                f"LiveSet.{element_string}.DeviceChain.AudioOutputRouting.LowerDisplayString",
            )
        out_target_element.set("Value", output_obj["target"])
        lower_display_string_element.set("Value", output_obj["lower_display_string"])
        logger.info("%sSet %s to %s", G, element_string, output_obj["lower_display_string"])

    def _parse_hex_path(self, text: str) -> Optional[str]:
        """Take raw hex string from XML entry and parses."""
        if not text:
            return None
        # Strip new lines and tabs from raw text to have one long hex string.
        abs_hash_path = text.replace("\t", "").replace("\n", "")
        byte_data = bytearray.fromhex(abs_hash_path)
        if byte_data[0:3] == b"\x00" * 3:  # Header only on mac projects.
            self.set_os = SetOperatingSystem.MAC_OS
            return utils.parse_mac_data(byte_data, abs_hash_path)
        else:
            self.set_os = SetOperatingSystem.WINDOWS_OS
            return utils.parse_windows_data(byte_data, abs_hash_path)

    def path_separator_type(self, path_str: str) -> str:
        """Get OS path string separator."""
        # TODO: Move this into utils.
        if "\\" in path_str:
            return "\\"
        elif "/" in path_str:
            return "/"
        else:
            raise ValueError(f"Couldn't parse OS path type! {path_str}")

    def search_plugins(self, plugin_name: str) -> Optional[pathlib.Path]:
        """Search for plugins and add them to self.found_vst_dirs."""
        if sys.platform == "win32":
            drive = os.environ["SYSTEMDRIVE"]
            _WINDOWS_VST3 = pathlib.Path(rf"{drive}\Program Files\Common Files\VST3")
            vst3_plugins = list(_WINDOWS_VST3.rglob("*.dll")) + list(_WINDOWS_VST3.rglob("*.vst3"))
            for vst3 in vst3_plugins:
                if plugin_name == vst3.name:
                    return vst3
            for directory in self.found_vst_dirs:
                for dll in directory.rglob("*.dll"):
                    # TODO rather match by MyPlugin.vst3/Contents/Resources/moduleinfo.json	?
                    # https://steinbergmedia.github.io/vst3_dev_portal/pages/Technical+Documentation/Locations+Format/Plugin+Format.html
                    if plugin_name == dll.name or plugin_name == dll.name.replace(".32", "").replace(".64", ""):
                        return dll
            return None
        else:

            # TODO match from sqlite db? (~/Library/Application\ Support/Ableton/Live\ Database )
            # TODO: custom folder plugin folder
            vst3_plugins = list(pathlib.Path("/Library/Audio/Plug-Ins/VST3").rglob("*.vst3/Contents/Info.plist")) + list(pathlib.Home() + pathlib.Path("/Library/Audio/Plug-Ins/VST3").rglob("*.vst3/Contents/Info.plist"))
            for vst3 in vst3_plugins:
                with open(vst3, 'rb') as file:
                    pl = plistlib.load(file)
                    if "CFBundleDisplayName" in pl:
                        if pl["CFBundleDisplayName"]== plugin_name:
                            return vst3.parent.parent
                    elif "CFBundleName" in pl:
                        if pl["CFBundleName"]== plugin_name:
                            return vst3.parent.parent
                    if "AudioComponents" in pl:
                        if pl["AudioComponents"][0]["name"] == plugin_name: #TODO: more than one component?
                            return vst3.parent.parent
                    elif plugin_name + ".vst3" == vst3.parent.parent.name:
                        return vst3.parent.parent

            # TODO do something with self.found_vst_dirs?

            return None

    # Plugin related functions.
    def parse_vst_element(
        self, vst_element: ET.Element
    ) -> Tuple[Optional[pathlib.Path], Optional[str], Optional[pathlib.Path]]:
        """Parse out VST element from vst xtree."""
        for plugin_path in ["Dir", "Path"]:
            path_results = vst_element.findall(f".//{plugin_path}")
            if len(path_results):
                if plugin_path == "Path":
                    if (full_path := path_results[0].get("Value")) is None:
                        logger.error("Couldn't get Path for %s", path_results[0])
                        continue
                    if not "/" in full_path and not "\\" in full_path:
                        if search_result := self.search_plugins(full_path):
                            return None, search_result.name, search_result
                        return None, full_path, None
                    path_separator = self.path_separator_type(full_path)
                    name = full_path.split(path_separator)[-1]
                    return pathlib.Path(full_path), name, None
                elif plugin_path == "Dir":
                    if (dir_bin := path_results[0].find("Data")) is None:
                        logger.error("Couldn't get Path for %s", path_results[0])
                        continue
                    if (text := dir_bin.text) is None:
                        continue
                    path = self._parse_hex_path(text)
                    name_ele = vst_element.find("FileName")
                    name = name_ele.get("Value", "") if name_ele is not None else "<>"
                    if not path:
                        logger.error("%sCouldn't parse absolute path for %s", Y, name)
                        return None, name, None
                    path_separator = self.path_separator_type(path)
                    if path[-1] == path_separator:
                        full_path = f"{path}{name}"
                    else:
                        full_path = f"{path}{path_separator}{name}"
                    return pathlib.Path(full_path), name, None

        logger.error("%sCouldn't parse plugin!", R)
        return None, None, None

    def list_plugins(self, vst_dirs: List[pathlib.Path]) -> List[pathlib.Path]:
        """Iterates through all plugin references and checks paths for VSTs."""
        self.found_vst_dirs.extend(vst_dirs)
        # TODO consider to log existing plugins as debug and only report missing ones by default as it is with the samples
        for plugin_element in self.root.iter("PluginDesc"):
            self.last_elem = plugin_element
            for vst_element in plugin_element.iter("VstPluginInfo"):
                full_path, name, potential = self.parse_vst_element(vst_element)
                exists = True if full_path and full_path.exists() else False
                if exists and full_path.parent not in self.found_vst_dirs:
                    self.found_vst_dirs.append(full_path.parent)
                elif not exists:
                    # Did not find plugin in saved path, try to search
                    potential = self.search_plugins(name)
                color = G if exists else R
                if potential and color == R:
                    color = Y
                logger.info(
                    "%sPlugin: %s, %sPlugin folder path: %s, %sExists: %s", color, name, M, full_path, color, exists
                )
                if potential:
                    logger.info("%s\tPotential alternative path for %s found: %s%s", CB, name, M, potential)
            for au_element in plugin_element.iter("AuPluginInfo"):
                name = au_element.find("Name").get("Value")
                manufacturer = get_element(plugin_element, "AuPluginInfo.Manufacturer", attribute="Value")
                logger.info(
                    "%sMac OS Audio Units are not saved with paths. Plugin %s: %s cannot be verified.",
                    M,
                    manufacturer,
                    name,
                )
                # TODO figure out how to match different name from components installed to stored set plugin.
                # au_components = pathlib.Path('/Library/Audio/Plug-Ins/Components').rglob('*.component')
            for vst3_element in plugin_element.iter("Vst3PluginInfo"):
                name = vst3_element.find("Name").get("Value")
                full_path = self.search_plugins(name)
                #manufacturer = "TODO: read from sqlite?" # get_element(plugin_element, "AuPluginInfo.Manufacturer", attribute="Value")
                exists = True if full_path and full_path.exists() else False
                color = G if exists else R
                logger.info(
                    "%sPlugin: %s, %sPlugin folder path: %s, %sExists: %s", color, name, M, full_path, color, exists
                )

        return self.found_vst_dirs

    def _list_samples(self) -> None:
        """Post Ableton 11 sample parser. Format changed from binary encoded paths to simple strings for all OSes."""
        missing_samples = 0
        for parsed in self._iterate_samples():
            if parsed.absolute_exists or parsed.relative_exists:
                # Sample will load in ableton, no need to do anything.
                logger.debug(
                    "%sSample %s found: Relative %s, Absolute %s",
                    G,
                    parsed.name,
                    parsed.relative,
                    parsed.absolute,
                )
                continue
            missing_samples += 1
            logger.warning(
                "%sSample %s missing: \n\tAbsolute[%s], Relative [%s]", R, parsed.name, parsed.absolute, parsed.relative
            )

    def list_samples(self) -> None:
        """Select correct sample parsing function."""
        if self.project_root_folder is None:
            self.find_project_root_folder()
        if self.version_tuple is None:
            self.load_version()
        if self.version_tuple is None:
            raise SetError("Version not parsed!")
        return self._list_samples()

    def _parse_samplepaths(
        self,
        absolute_path: Optional[pathlib.Path],
        relative_path: Optional[pathlib.Path],
        verbose: bool,
        saved_filesize: int,
    ) -> bool:
        absolute_found = absolute_path is not None and absolute_path.exists()
        relative_found = relative_path is not None and relative_path.exists()
        if not absolute_found and not relative_found:
            if absolute_path and absolute_path not in self.missing_absolute_samples:
                self.missing_absolute_samples.append(absolute_path)
            if relative_path and relative_path not in self.missing_relative_samples:
                self.missing_relative_samples.append(relative_path)
            return False
        if absolute_found and absolute_path is not None:
            local_filesize = absolute_path.stat().st_size
            if verbose:
                size_match = saved_filesize == local_filesize
                logger.info(
                    "%sAbsolute path sample found: %s\n\tFile size %s matches saved filesize %s: %s%s",
                    G,
                    absolute_path,
                    local_filesize,
                    saved_filesize,
                    G if size_match else R,
                    size_match,
                )
        if relative_found and relative_path is not None:
            local_filesize = relative_path.stat().st_size
            size_match = saved_filesize == local_filesize
            if verbose:
                logger.info(
                    "%sRelative(collect and save) sample found: %s\n\tFile size %s matches saved filesize %s: %s%s",
                    G,
                    relative_path,
                    local_filesize,
                    saved_filesize,
                    G if size_match else R,
                    size_match,
                )
        if absolute_found or relative_found:
            return True
        return False

    def sample_results(self, missing_samples: int) -> None:
        """logger.info results of sample search."""
        color = G if not missing_samples else Y
        logger.info(
            "%sTotal missing sample references: %s%s%s, this can include duplicate references to the same sample so "
            "only unique paths are listed here. Relative paths are created using collect-and-save. If either sample "
            "path is found Ableton will load the sample.",
            color,
            M,
            missing_samples,
            color,
        )
        if self.missing_relative_samples:
            rel_string = "\n\t".join((str(x) for x in self.missing_relative_samples))
            logger.info("%sMissing Relative paths:%s\n\t%s", Y, R, rel_string)
        if self.missing_absolute_samples:
            abs_string = "\n\t".join((str(x) for x in self.missing_absolute_samples))
            logger.info("%sMissing Absolute paths:%s\n\t%s", Y, R, abs_string)

    def _iterate_samples(self) -> Generator[utils.SampleRef, None, None]:
        """Iterate through set sample references and build sample list."""
        if self.sample_list:
            for parsed in self.sample_list:
                yield parsed
            return
        for sample_ref in self.root.iter("SampleRef"):
            parsed = utils.SampleRef.from_element(sample_ref, self.version_tuple, self.project_root_folder)
            self.sample_list.append(parsed)
            yield parsed
        return

    def fix_samples(self, db: Dict[str, Dict[str, str]], collect_and_save: bool = False, force: bool = False) -> bool:
        """Fix broken sample paths.

        Args:
            db: database loaded from json.
            collect_and_save: copy any found samples into the project folder, the same as ableton's collect
                and save
            force: used with collect_and_save. When the same name sample is found in the project, force replace
                it if the project's current file is a different file size.
        """
        self.find_project_root_folder()
        found_samples: Dict[str, Dict[str, str]] = {}
        missing_samples = 0
        fixed_samples = 0
        skip_search = False
        for parsed in self._iterate_samples():
            if parsed.absolute_exists or parsed.relative_exists:
                # Sample will load in ableton, no need to do anything.
                continue
            missing_samples += 1

            # Skip builtin pack content for now. Can revisit this later but these samples probably will fix
            # automatically in ableton on set load.
            factory_packs = ["/Resources/Builtin/Samples", "Ableton/Factory Packs"]
            if any([x in str(parsed.absolute.parent) for x in factory_packs]):
                logger.debug("%sSkipping builtin pack content: %s", Y, parsed.absolute)
                continue

            # There's often the same sample referenced many times in the same set, check previous found first.
            for smp_path, smp_info in found_samples.items():
                if self._fix_sample(collect_and_save, parsed, smp_info, smp_path, found_samples, force):
                    fixed_samples += 1
                    skip_search = True
                    break
            if skip_search:
                skip_search = False
                continue
            # Iterating through hashes is extremely fast :D
            for smp_path, smp_info in db.items():
                if self._fix_sample(collect_and_save, parsed, smp_info, smp_path, found_samples, force):
                    fixed_samples += 1
                    break
            else:
                logger.warning(
                    "%sCould not find sample for %s\n%s\n%s", Y, parsed.name, parsed.absolute, parsed.relative
                )

        logger.info(
            "%sOrignal missing sample count: %s, Samples fixed: %s, Couldn't fix: %s",
            G if fixed_samples == missing_samples else R,
            missing_samples,
            fixed_samples,
            missing_samples - fixed_samples,
        )

    def _fix_sample(
        self,
        collect_and_save: bool,
        parsed: utils.SampleRef,
        smp_info: Dict[str, str],
        smp_path: str,
        found_samples: Dict[str, Dict[str, str]],
        force: bool,
    ) -> bool:
        """Attempt to fix sample if matches DB entry.

        size is not always stored in ableton sets unfortunately, but we do usually have last_modified.
        This is not perfect, but the probability of a file name matching and it's last modification time
        matching and being a false positive are quite low.
        """
        if smp_info.get("name") != parsed.name:
            return False
        size_match = parsed.size and smp_info.get("size") == parsed.size
        modified_match = parsed.last_modified and parsed.last_modified == int(smp_info.get("last_modified"))
        if not size_match and not modified_match:
            return False

        logger.debug("\n\n%sFound potential match %s, \n[%s]\n%s%s", G, smp_path, smp_info, M, parsed)
        found_samples[smp_path] = smp_info
        replacement_sample = pathlib.Path(smp_path)

        if collect_and_save and self.project_root_folder:
            # Relative type 3 is collected and saved, 1 is absolute path.
            relative_type = parsed.get_relative_type()
            if relative_type == 3:
                rel_path = str(parsed.get_relative_value())
            else:
                rel_path = "Samples/Imported"
            (self.project_root_folder / rel_path).mkdir(parents=True, exist_ok=True)

            copied_sample = self.project_root_folder / rel_path / smp_info.get("name")
            if copied_sample.exists() and copied_sample.stat().st_size != parsed.size:
                logger.error(
                    "%sCannot copy sample %s, would replace existing one in project with " "same name! Skipping...",
                    R,
                    copied_sample,
                )
                return False
            elif copied_sample.exists() and copied_sample.stat().st_size == parsed.size:
                pass
            else:
                shutil.copy(replacement_sample, copied_sample)
            parsed.set_relative(f"{rel_path}/{copied_sample.name}")
            parsed.set_relative_type(3)
        elif collect_and_save and not self.project_root_folder:
            logger.warning(
                "%sProject root () not found, can't collect and save this sample. " "Using absolute path instead..", Y
            )
            parsed.set_absolute(replacement_sample)
            parsed.set_relative_type(1)
        else:
            parsed.set_absolute(replacement_sample)
            parsed.set_relative_type(1)
        return True

    def gradient_tracks(self) -> None:
        """Make a rough gradient across tracks using built in colors."""
        if not self.tracks:
            self.load_tracks()
        for clr_ind, track in zip(color_tools.create_gradient_ableton(len(self.tracks)), self.tracks):
            track.color = clr_ind

            clipview_clr_elements = list(track.clip_clipview_colors())
            clip_view_gradient = color_tools.create_gradient_ableton(len(clipview_clr_elements), starting_index=clr_ind)
            for sub_ind, clip_clr_ele in zip(clip_view_gradient, clipview_clr_elements):
                clip_clr_ele.set("Value", str(sub_ind))

            arangement_clr_elements = list(track.clip_arangement_colors())
            clip_view_gradient = color_tools.create_gradient_ableton(
                len(arangement_clr_elements), starting_index=clr_ind
            )
            for sub_ind, clip_clr_ele in zip(clip_view_gradient, arangement_clr_elements):
                clip_clr_ele.set("Value", str(sub_ind))

    def trim_drum_racks(self, drum_track_ids: list[str])-> None:
        """Remove all chains from all drum racks on the given tracks that don't have any active notes in the session arrangement clips"""
        if len(drum_track_ids) == 1 and drum_track_ids[0] == "all":
            track_ids = set()
            for drums in self.root.findall(".//DrumGroupDevice"):
                track_ids.add(find_parent(self.root, drums, "MidiTrack").get("Id"))
            self.trim_drum_racks(list(track_ids))
            return
        for id in drum_track_ids:
            self._trim_drum_rack(id)

    def _trim_drum_rack(self, drum_track_id: list[str])-> None:
        track_found = False
        for track in self.tracks:
            if track.id == drum_track_id:
                track_found = True
                logger.info("%sTrimming drum rack(s) on track %s", C, drum_track_id)
                drum_groups = track.track_root.findall("DeviceChain//DrumGroupDevice")
                if len(drum_groups) == 0:
                    logger.error("%sNo drum rack(s) found on track %s", R ,drum_track_id)
                    break
                # using ../ is a workaround due to restrictions in the filter predicate which cannot select based on child element attributes
                clips = track.track_root.findall("DeviceChain//MidiClip/Disabled[@Value='false']/..")
                played_notes =  self._get_unique_notes(clips)
                removed_something = False
                for drum_group in drum_groups:
                    group_name = drum_group.find("UserName").get("Value")
                    if not group_name:
                        group_name = "UNKNOWN"
                    branches_to_remove = self._get_unused_drum_branches(drum_group, played_notes)
                    branch_container = drum_group.find("Branches")
                    for id in branches_to_remove:
                        branch = branch_container.find(f"DrumBranch[@Id='{id}']")
                        chain_name = branch.find("Name/EffectiveName").get("Value")
                        logger.info("%sRemoving unsused chain %s%s%s %s%s", C, G, group_name, C, R, chain_name)
                        branch_container.remove(branch)
                        removed_something = True
                break #track was found
                              
        if not track_found:
            logger.error("%sTrack %s was not found", R, drum_track_id)
        elif not removed_something:
            logger.info("%sTrack %s has no unused chains", C, drum_track_id)
    
    def _get_unused_drum_branches(self, drum_group: ET.Element, played_notes: set[int]) -> list[ET.Element]:
        """Find all DrumBranch elements (aka Drum Rack Chains) that are receiving notes other than the ones given."""
        unsued_branch_ids = []
        branches = drum_group.findall("Branches/DrumBranch")
        for branch in branches:
            receivers = branch.findall("BranchInfo/ReceivingNote")
            for receiver in receivers:
                #note 128 is "All"
                note = 128 - int(receiver.get("Value"))
                if note != 128 and not note in played_notes:
                    unsued_branch_ids.append(branch.get("Id"))
        return unsued_branch_ids

    def _get_unique_notes(self, clips: list[ET.Element]) -> set[int]:
        """Find all MIDI notes that are played by any of the given clips elements"""
        played_notes = set[int]()
        for clip in clips:
            midi_keys = clip.findall(".//MidiNoteEvent[@IsEnabled='true']../../MidiKey")
            for key in midi_keys:
                played_notes.add(int(key.get("Value")))  
        return played_notes

    def split_midi_tracks(self, midi_track_ids: list[str]) -> None:
        for id in midi_track_ids:
            self._split_midi_track(id)

    def _split_midi_track(self, midi_track_id: str) -> None:
        track_found = False
        splitted_something = False
        processed_groups = 0
        for track in self.tracks:
            if track.id == midi_track_id and track.type == "MidiTrack":
                track_found = True
                logger.info("%sSplitting MIDI track %s", C, midi_track_id)
                self._process_midi_split(track)
                break #track was found
        if not track_found:
            logger.error("%sTrack %s was not found or is not a MIDI track", R, midi_track_id)

    def _process_midi_split(self, track: AbletonTrack):
        logger.debug("/// MIDI split -- on track "+track.id+" "+track.name)
        clips = track.track_root.findall("DeviceChain//MidiClip/Disabled[@Value='false']/..")
        played_notes =  self._get_unique_notes(clips)
        note_list = list(played_notes)
        if len(note_list) > 1:
            logger.debug(note_list)
            new_track = self._duplicate_track(track)
            self._clear_notes(track, [note_list[0]]) #preserve only the first note on current track
            self._clear_notes(new_track, note_list[1::]) # preseve all but the first note on new track
            logger.debug(f"recursing into new track {new_track.id}")
            self._process_midi_split(new_track) #repeat the process on new track
        else:
            logger.debug(f"This track is OK: {len(note_list)} note(s)")

    def split_drum_racks(self, drum_track_ids: list[str])-> None:
        if len(drum_track_ids) == 1 and drum_track_ids[0] == "all":
            track_ids = set()
            for drums in self.root.findall(".//DrumGroupDevice"):
                track_ids.add(find_parent(self.root, drums, "MidiTrack").get("Id"))
            self.split_drum_racks(list(track_ids))
            return
        for id in drum_track_ids:
            self._split_drum_rack(id)

    def _split_drum_rack(self, drum_track_id: str)-> None:
        track_found = False
        for track in self.tracks:
            if track.id == drum_track_id:
                track_found = True
                logger.info("%sSplitting drum rack(s) on track %s", C, drum_track_id)
                if len(track.track_root.findall(".//DrumBranch")) == 0:
                    logger.error("%sNo drum rack chains found on track %s", R ,drum_track_id)
                    break
                else:
                    self._process_drum_split(track)
                break #track was found
        if not track_found:
            logger.error("%sTrack %s was not found", R, drum_track_id)

    def _process_drum_split(self, track: AbletonTrack):
        logger.debug("///----- on track "+track.id+" "+track.name)
        drum_groups = track.track_root.findall("DeviceChain//DrumGroupDevice")
        if len(drum_groups) == 0:
            #delete track without any drum groups (expected to arrive here in recursion)
            logger.debug("delete track without groups")
            find_parent(self.root, track.track_root).remove(track.track_root)
            self.tracks.remove(track)
        elif len(drum_groups) > 0:
            drum_group = drum_groups[0]
            branch_container = drum_group.find("Branches")
            branches = branch_container.findall("DrumBranch")
            if len(branches) == 0:
                # the group is empty, so we remove it and then start over again to get to the next group (if any)
                logger.debug("first group is empty, deleting it and re-starting")
                find_parent(track.track_root, drum_group).remove(drum_group)
                self._process_drum_split(track)
            elif len(branches) > 1:
                branch = branches[0]
                branch_name = branch.find("Name/UserName").get("Value")
                if not branch_name:
                    branch_name = branch.find("Name/EffectiveName").get("Value")
                logger.debug(f"branch {branch_name}")
                branch.set("_copyHelper","1") 
                new_track = self._duplicate_track(track)
                branch.attrib.pop("_copyHelper") #will need this on the duplicated track only

                logger.debug(f"relabeling this track to {branch_name}")
                track.name = branch_name
                track.track_root.find("Name/UserName").set("Value", branch_name)
                
                new_track_root = new_track.track_root
                
                to_remove = new_track_root.find(f"DeviceChain//DrumBranch[@_copyHelper='1']")
                to_remove_from = find_parent(new_track_root, to_remove)
                to_remove_from.remove(to_remove)

                logger.debug(f"removing {len(branches[1::])} remaining branches from track {track.id}")
                for branch in branches[1:]:
                    find_parent(track.track_root, branch).remove(branch)

                logger.debug(f"removing {len(drum_groups[1::])} remaining groups from track {track.id}")
                for group in drum_groups[1::]:
                    find_parent(track.track_root, group).remove(group)

                logger.debug("cleaning up instrument rack chains")
                instr_group = find_parent(track.track_root, drum_group, "InstrumentGroupDevice")
                if instr_group is not None:
                    logger.debug(f"instrument rack found {instr_group.tag} {instr_group.get('Id')}")
                    #now, remove all intrunment chains that contain drum racks (except the one)
                    instr_branch_container = instr_group.find("Branches")
                    instr_branches = instr_branch_container.findall("InstrumentBranch")
                    logger.debug(f"instrument rack contains {len(instr_branches)} branches")
                    for instr_branch in instr_branches:
                        devices = instr_branch.findall("DeviceChain//Devices/*")
                        # TODO there could be devices like EQ left which make no sense if the drums are gone
                        # and we can be pretty sure that there was no other midi-to-audio device here, right?
                        if len(devices) == 0: #devices is empty
                            logger.debug(f"removing empty instrument branch {instr_branch.get('Id')}")
                            instr_branch_container.remove(instr_branch)
                        else:
                            logger.debug(f"instrument chain {instr_branch.get('Id')} kept due to {len(devices)} devices:")
                            logger.debug(f"    {devices}")
                else:
                    logger.debug("No instrument rack found")


                self._clear_ineffective_drum_notes(track)
                logger.debug(f"recursing into new track {new_track.id}")
                self._process_drum_split(new_track)
            else:
                #only one chain
                logger.debug("---- reached single-chain group")
                branch = branches[0]
                branch_name = branch.find("Name/UserName").get("Value")
                if not branch_name:
                    branch_name = branch.find("Name/EffectiveName").get("Value")
                
                logger.debug(f"relabeling to {branch_name}")
                track.name = branch_name
                track.track_root.find("Name/UserName").set("Value", branch_name)

                drum_group.set("_copyHelper","HELLO")
                new_track = self._duplicate_track(track)
                drum_group.attrib.pop("_copyHelper")
        
                new_track_root = new_track.track_root
            
                to_remove = new_track_root.find(f"DeviceChain//DrumGroupDevice[@_copyHelper='HELLO']")
                to_remove_from = find_parent(new_track_root, to_remove)
                to_remove_from.remove(to_remove)

                logger.debug(f"removing {len(drum_groups[1::])} remaining groups from track {track.id}")
                for group in drum_groups[1::]:
                    find_parent(track.track_root, group).remove(group)

                self._clear_ineffective_drum_notes(track)

                logger.debug("::cleaning up instrument rack chains")
                drum_group.set("_lookupHelper","1")
                is_in_instr_group = track.track_root.find("DeviceChain/DeviceChain/Devices//InstrumentGroupDevice//DrumGroupDevice[@_lookupHelper='1']")
                drum_group.attrib.pop("_lookupHelper","1")
                if is_in_instr_group:
                    #This drum rack is part of an instrument rack
                    parent = drum_group
                    found_instr_group = False
                    while parent.tag != "MidiTrack":
                        parent = find_parent(track.track_root, parent)
                        if parent.tag == "InstrumentGroupDevice":
                            found_instr_group = True
                            #we have found the intrument rack
                            logger.debug(f"instrument rack found {parent.tag}  {parent.get('Id')}")
                            #now, remove all intrunment chains that contain drum racks (except the one)
                            instr_branch_container = parent.find("Branches")
                            instr_branches = instr_branch_container.findall("InstrumentBranch")
                            logger.debug(f"instrument rack contains {len(instr_branches)} branches")
                            for instr_branch in instr_branches:
                                devices = instr_branch.findall("DeviceChain//Devices/*")
                                # TODO there could be devices like EQ left which make no sense if the drums are gone
                                # and we can be pretty sure that there was no other midi-to-audio device here, right?
                                if len(devices) == 0: #devices is empty
                                    logger.debug(f"removing empty instrument branch {instr_branch.get('Id')}")
                                    instr_branch_container.remove(instr_branch)
                                else:
                                    logger.debug(f"instrument chain {instr_branch.get('Id')} kept due to {len(devices)} devices:")
                                    logger.debug(f"    {devices}")
                            break #stop walking up
                    if not found_instr_group:
                        logger.debug("No instrument rack found")

                logger.debug(f"recursing into new track {new_track.id}")
                self._process_drum_split(new_track)
  
    def _clear_ineffective_drum_notes(self, track: AbletonTrack):
        receivers = track.track_root.findall(".//DrumBranch/BranchInfo/ReceivingNote")
        received_notes = set[int]()
        for receiver in receivers:
            note = 128 - int(receiver.get("Value"))
            #note 128 is "All"
            if note == 128:
                logger.debug("there's a chain receiving all notes, clearing notes in clips makes no sense")
                return
            note = 128 - int(receiver.get("Value"))
            received_notes.add(note)
        self._clear_notes(track, received_notes)  

    def _clear_notes(self,track: AbletonTrack, notes_to_preserve: list[int]):
        logger.debug(f"preseve: {notes_to_preserve}")
        clips = track.track_root.findall(".//MidiClip/Disabled[@Value='false']/..")
        for clip in clips:
            self._clear_clip_notes(clip, notes_to_preserve)
             
    def _clear_clip_notes(self, clip: ET.Element, notes_to_preserve: list[int]):
        midi_keys = clip.findall(".//MidiNoteEvent[@IsEnabled='true']../../MidiKey")
        for key in midi_keys:
            midi_note = int(key.get("Value"))
            if not midi_note in notes_to_preserve:
                logger.debug(f"removing MIDI note {midi_note}")
                key_track = find_parent(clip, key, "KeyTrack")
                find_parent(clip, key_track).remove(key_track)
        #TODO need to fix Ids of KeyTrack elements?

    def _fixPointees(self, track: AbletonTrack):
        pointee_element = self.root.find("LiveSet/NextPointeeId")
        next_pointee_id = int(pointee_element.get("Value"))
        new_track_element = track.track_root

        pointees = new_track_element.findall(".//Pointee") \
            + new_track_element.findall(".//AutomationTarget") \
            + new_track_element.findall(".//ModulationTarget") \
            + new_track_element.findall(".//FluxModulationTarget") \
            + new_track_element.findall(".//GrainSizeModulationTarget") \
            + new_track_element.findall(".//SampleOffsetModulationTarget") \
            + new_track_element.findall(".//TranspositionModulationTarget") \
            + new_track_element.findall(".//VolumeModulationTarget") \
            + new_track_element.findall(".//MidiControllers")

        for p in pointees:
            if p.tag == "MidiControllers":
                for e in p:
                    if e.tag.startswith("ControllerTargets."):
                        new_id = str(next_pointee_id)
                        self._fixSinglePointee(new_track_element, e, new_id)
                        next_pointee_id += 1
            else:
                new_id = str(next_pointee_id)
                self._fixSinglePointee(new_track_element, p, new_id)
            next_pointee_id += 1
        pointee_element.set("Value", str(next_pointee_id))


    def _fixSinglePointee(self, new_track_element,  e, new_id):
        old_id = e.get("Id")
        e.set("Id", new_id)
        automations = new_track_element.findall(".//EnvelopeTarget/PointeeId[@Value='"+old_id+"']")
        for automation in automations:
            automation.set("Value", new_id)

    def _duplicate_track(self, track: AbletonTrack) -> AbletonTrack:
        
        new_track_id = self._get_max_track_id() + 1
        logger.debug(f"duplicating track {track.id} to track {new_track_id}")
        
        new_track_element = copy.deepcopy(track.track_root)
        new_track_element.set("Id", str(new_track_id))
        
        new_track = AbletonTrack(new_track_element, self.version_tuple)
        self._fixPointees(new_track)

        device_selectors = new_track_element.findall("SelectedDevice")
        for selector in device_selectors:
            device_selectors.set("Value", "0")
        device_selectors = new_track_element.findall("SelectedEnvelope")
        for selector in device_selectors:
            device_selectors.set("Value", "0")
        device_selectors = new_track_element.findall("IsSelected")
        for selector in device_selectors:
            device_selectors.set("Value", "false")
        
        track_container = self.root.find("LiveSet/Tracks")
        new_position = list(track_container).index(track.track_root) + 1
        track_container.insert(new_position, new_track_element)
        self.tracks.insert(new_position, new_track)

        return new_track

    def _get_max_track_id(self) -> int:
        max_id = 0
        all_tracks = self.root.findall("LiveSet/Tracks/*")
        for track in all_tracks:
            max_id = max(max_id, int(track.get("Id")))
        return max_id

    def number_tracks(self):
        """Ensure all tracks will show a sequence number in their name. Will set the UserName to start with '#-'. Tracks that start with '_' are ignored."""
        logger.info("%s Numbering tracks", C)
        for track in self.tracks:
            if track.type == "ReturnTrack":
                continue
            user_name_element = track.track_root.find("Name/UserName")
            user_name = user_name_element.get("Value")
            if not user_name.startswith("#") and not user_name.startswith("_"):
                base_name = user_name
                if not base_name:
                    base_name = track.track_root.find("Name/EffectiveName").get("Value")
                new_base_name = re.sub("^[0-9]+-? *", "", base_name)
                base_name = new_base_name
                new_name =f"#-{base_name}"
                user_name_element.set("Value", new_name)
                track.name =  new_name

    def show_main_notes(self):
        info_text = self.root.find(f"LiveSet/{self.mainId}Track/Name/Annotation").get("Value")
        if info_text:
            logger.info("%s%s track info text: %s%s",C, self.mainId, G, info_text)
        else:
            logger.info("%s%s track has no info text.",Y, self.mainId)
    
    def sort_by_arrangement(self):
        """Sort tracks by their earliest arrangement clip start time"""
        logger.info("%sSorting tracks by arrangement clip start time", C)
        track_starts = {}
        tracks_by_id = {}
        group_tracks = []
        for track in self.tracks:
            if track.type == "ReturnTrack":
               continue
            if track.type == "GroupTrack":
                group_tracks.append(track.id)
                tracks_by_id[track.id] = track
                continue
            tracks_by_id[track.id] = track
            arrangement_clips = track.track_root.findall(".//ClipTimeable/ArrangerAutomation/Events/MidiClip")
            earliest_start = float('inf')
            for clip in arrangement_clips:
                earliest_start = min(earliest_start, int(clip.get("Time")))
            track_starts[track.id] = earliest_start

            if track.group_id != "-1":
                group_start = track_starts.get(track.group_id)
                if group_start is None:
                    group_start = float('inf')
                group_start = min(group_start, earliest_start)
                track_starts[track.group_id] = group_start

        # order from last to first ocurrence so we can work backwards and always push tracks to the front with insert(0, track)
        # inner sort will bring group tracks to the front so that the group's tracks will be after the group track when sorted by start time
        order = reversed(sorted(sorted(track_starts.items(), key=lambda x : float('inf') if x[1] == -1 else x[1]), key=lambda x : 0 if x[0] in group_tracks else 1))

        print(track_starts)

        # apply order by time to all tracks
        for (track_id, _) in order:
            print(track_id, track_starts[track_id])
            track = tracks_by_id[track_id]
            self.tracks.remove(track)
            self.tracks.insert(0, track)

        self.print_tracks()

        print()    
        # pull tracks into their groups again
        new_order = []
        groups = {}
        for track in self.tracks:
            trackInfo = None
            if track.type == "GroupTrack":
                trackInfo = {"type": "group", "track": track, "children":[]}
                groups[track.id] = trackInfo
                if track.group_id != "-1":
                    groups[track.group_id]["children"].append(trackInfo)
                else:
                    new_order.append(trackInfo)
            else:
                trackInfo = {"type": "track", "track": track}
                if not track.group_id in groups:
                    print("group "+track.group_id+" not available yet")
                if track.group_id != "-1":
                    groups[track.group_id]["children"].append(trackInfo)
                else:
                    new_order.append(trackInfo)

        sorted_track_list = []
        for trackInfo in new_order:
            self._append_recursive(sorted_track_list, trackInfo)

        track_container = self.root.find("LiveSet/Tracks")
        self.tracks = sorted_track_list

        #finally apply the order to XML elements
        for t in reversed(self.tracks):
            track_container.remove(t.track_root)
            track_container.insert(0, t.track_root)
        
        
    def _append_recursive(self, sorted_track_list: list[AbletonTrack], trackInfo: dict):
        sorted_track_list.append(trackInfo["track"])
        if trackInfo["type"] == "group":
            for child in trackInfo["children"]:
                self._append_recursive(sorted_track_list, child)

 




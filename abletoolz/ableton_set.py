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
from abletoolz.misc import CB, RB, RST, STEREO_OUTPUTS, B, C, G, M, R, Y, get_element

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

    def save_set(self, append_bars_bpm: bool = False, prepend_version: bool = False) -> None:
        """Save set to disk with optional filename modifications.

        This function saves the current set to disk, first creating a backup of the original file.
        It optionally appends the number of bars and BPM to the filename, and/or prepends the version number.
        The actual writing of the set to disk is performed in a separate thread.

        Args:
            append_bars_bpm: If True, append the number of bars and BPM to the filename.
            prepend_version: If True, prepend the version number to the filename.
        """
        utils.create_backup(self.path)
        if append_bars_bpm:
            cleaned_name = re.sub(r"_\d{1,3}bars_\d{1,3}\.\d{2}bpm", "", self.path.stem)
            new_filename = cleaned_name + f"_{self.furthest_bar}bars_{self.bpm:.2f}bpm.als"
            self.path = pathlib.Path(self.path.parent / new_filename)
            logger.debug("%sAppending bars and bpm, new set name: %s.als", M, self.path.stem)

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

    def print_tracks(self) -> None:
        """logger.infos track info."""
        logger.info("Tracks:\n%s", "\n".join([str(x) for x in self.tracks]))

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
        post_10_bpm = "LiveSet.MasterTrack.DeviceChain.Mixer.Tempo.Manual"
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
                f"LiveSet.{element_string}.MasterChain.AudioOutputRouting.Target",
            )
            lower_display_string_element = get_element(
                self.root,
                f"LiveSet.{element_string}.MasterChain.AudioOutputRouting.LowerDisplayString",
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

            # TODO match from sqlite db?
            vst3_plugins = list(pathlib.Path("/Library/Audio/Plug-Ins/VST3").rglob("*.vst3/Contents/Info.plist")) # TODO: ~/Library and custom folder + list(pathlib.Path("/Library/Audio/Plugins/VST3").rglob("*.vst3"))
            for vst3 in vst3_plugins:
                #print(vst3)
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
                            print("using AudioComponents[0]")
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
            #print(ET.tostring(plugin_element))
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
                #if potential and color == R:
                #    color = Y
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

    def trim_drum_racks(self, drum_track_ids)-> None:
        for id in drum_track_ids:
            self.trim_drum_rack(id)

    def trim_drum_rack(self, drum_track_id)-> None:
        """Remove all chains from all drum racks on the given track that don't have any active notes in the session arrangement clips"""
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
                    group_name = drum_group.find("UserName").get("Value") # TODO alternate name    
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
            logger.info("%sTrack %s has no unused chains", G, drum_track_id)
    
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

    def split_drum_racks(self, drum_track_ids)-> None:
        for id in drum_track_ids:
            self.split_drum_rack(id)

    def split_drum_rack(self, drum_track_id)-> None:
        track_found = False
        splitted_something = False
        processed_groups = 0
        for track in self.tracks:
            if track.id == drum_track_id:
                track_found = True
                logger.info("%sSplitting drum rack(s) on track %s", C, drum_track_id)
                if len(track.track_root.findall(".//DrumBranch")) == 0:
                    logger.error("%sNo drum rack chains found on track %s", R ,drum_track_id)
                    break
                else:
                    self._process(track)
                break #track was found
        if not track_found:
            logger.error("%sTrack %s was not found", R, drum_track_id)
        # elif not splitted_something:
        #     logger.info("%sTrack %s had no chains to extract", G, drum_track_id)

    def _process(self, track: AbletonTrack):
        print ("///----- on track "+track.id+" "+track.name)
        drum_groups = track.track_root.findall("DeviceChain//DrumGroupDevice")
        print (str(len(drum_groups))+" drum groups")
        if len(drum_groups) == 0:
            #delete track without any drum groups (expected to arrive here in recursion)
            print("delte track without groups")
            self.find_parent(self.root, track.track_root).remove(track.track_root)
        elif len(drum_groups) > 0:
            drum_group = drum_groups[0]
            branch_container = drum_group.find("Branches")
            branches = branch_container.findall("DrumBranch")
            if len(branches) == 0:
                # the group is empty, so we remove it and then start over again to get to the next group (if any)
                print("first group is empty, deleting it and re-starting")
                self.find_parent(track.track_root, drum_group).remove(drum_group)
                self._process(track)
            elif len(branches) > 1:
                branch = branches[0]
                branch_name = branch.find("Name/UserName").get("Value")
                if not branch_name:
                    branch_name = branch.find("Name/EffectiveName").get("Value")
                print("branch " +branch_name)
                branch.set("_copyHelper","HELLO")
                new_track = self._duplicate_track(track)
                branch.attrib.pop("_copyHelper")

                print("relabeling this track to" +branch_name)
                track.name = branch_name
                track.track_root.find("Name/UserName").set("Value", branch_name)
                
                new_track_root = new_track.track_root
                
                to_remove = new_track_root.find(f"DeviceChain//DrumBranch[@_copyHelper='HELLO']")
                to_remove_from = self.find_parent(new_track_root, to_remove)
                #to_remove_from = new_track.find(f"DeviceChain//DrumBranch[@_copyHelper='HELLO']/..")
                to_remove_from.remove(to_remove)

                print("removing "+str(len(branches[1::]))+" remaining branches from track "+ track.id)
                for branch in branches[1:]:
                    self.find_parent(track.track_root, branch).remove(branch)

                print("removing "+str(len(drum_groups[1::]))+" remaining groups from track "+ track.id)
                for group in drum_groups[1::]:
                    self.find_parent(track.track_root, group).remove(group)

                print("::cleaning up instrument rack chains")
                drum_group.set("_lookupHelper","1")
                is_in_instr_group = track.track_root.find("DeviceChain/DeviceChain/Devices//InstrumentGroupDevice//DrumGroupDevice[@_lookupHelper='1']")
                if is_in_instr_group:
                    #This drum rack is part of an instrument rack
                    parent = drum_group
                    found_instr_group = False
                    while parent.tag != "MidiTrack":
                        parent = self.find_parent(track.track_root, parent)
                        if parent.tag == "InstrumentGroupDevice":
                            found_instr_group = True
                            #we have found the intrument rack
                            print("instrument rack found " + parent.tag +" " + parent.get("Id"))
                            #now, remove all intrunment chains that contain drum racks (except the one)
                            instr_branch_container = parent.find("Branches")
                            instr_branches = instr_branch_container.findall("InstrumentBranch")
                            print("instrument rack contains "+str(len(instr_branches))+" branches")
                            for instr_branch in instr_branches:
                                devices = instr_branch.findall("DeviceChain//Devices/*")
                                # TODO there could be devices like EQ left which make no sense if the drums are gone
                                # and we can be pretty sure that there was no other midi-to-audio device here, right?
                                if len(devices) == 0: #devices is empty
                                    print("__________ removing empty instrument branch "+instr_branch.get("Id"))
                                    instr_branch_container.remove(instr_branch)
                                else:
                                    print("instrument chain "+instr_branch.get("Id")+" kept due to "+str(len(devices))+" devices:")
                                    print("    "+str(devices))
                            break #stop walking up
                    if not found_instr_group:
                        print("No instrument rack found")


                

                drum_group.attrib.pop("_lookupHelper","1")
                print("recursing into new track "+new_track.id)
                self._process(new_track)
            else:
                #only one chain
                print("---- reached single-chain group")
                branch = branches[0]
                branch_name = branch.find("Name/UserName").get("Value")
                if not branch_name:
                    branch_name = branch.find("Name/EffectiveName").get("Value")
                
                print("relabeling to "+branch_name)
                track.name = branch_name
                track.track_root.find("Name/UserName").set("Value", branch_name)

                drum_group.set("_copyHelper","HELLO")
                new_track = self._duplicate_track(track)
                drum_group.attrib.pop("_copyHelper")
        
                new_track_root = new_track.track_root
            
                to_remove = new_track_root.find(f"DeviceChain//DrumGroupDevice[@_copyHelper='HELLO']")
                to_remove_from = self.find_parent(new_track_root, to_remove)
                to_remove_from.remove(to_remove)

                print("removing "+str(len(drum_groups[1::]))+" remaining groups from track "+ track.id)
                for group in drum_groups[1::]:
                    self.find_parent(track.track_root, group).remove(group)

                print("recursing into new track "+new_track.id)
                self._process(new_track)
  

    def find_parent(self, root, element):
        for parent in root.iter():
            for child in parent:
                if child is element:
                    return parent

    def _duplicate_track(self, track: AbletonTrack) -> AbletonTrack:
        
        new_track_element = copy.deepcopy(track.track_root)
        new_track_element.set("Id",str(self._get_max_track_id() + 1))

        print ("duplicating track "+track.id+" to track "+new_track_element.get("Id"))
        
        pointee_element = self.root.find("LiveSet/NextPointeeId")
        next_pointee_id = int(pointee_element.get("Value"))
        #print(next_pointee_id)
        pointees = new_track_element.findall(".//PointeeId") \
            + new_track_element.findall(".//AutomationTarget") \
            + new_track_element.findall(".//ModulationTarget") \
            + new_track_element.findall(".//FluxModulationTarget") \
            + new_track_element.findall(".//GrainSizeModulationTarget") \
            + new_track_element.findall(".//SampleOffsetModulationTarget") \
            + new_track_element.findall(".//TranspositionModulationTarget") \
            + new_track_element.findall(".//VolumeModulationTarget") \
            + new_track_element.findall(".//EnvelopeTarget")
        
        #print(str(len(pointees))+" pointees in new track")
        for p in pointees:
            if p.tag == "PointeeId":
                p.set("Value", str(next_pointee_id))
            else:
                p.set("Id", str(next_pointee_id))
            next_pointee_id += 1
            # print(next_pointee_id)
        
        pointee_element.set("Value", str(next_pointee_id))
        
        track_container = self.root.find("LiveSet/Tracks")
        new_position = list(track_container).index(track.track_root) + 1


        #new_position = len(self.root.findall("LiveSet/Tracks/*")) - len(self.root.findall("LiveSet/Tracks/ReturnTrack")) 
        
        track_container.insert(new_position, new_track_element)
        new_track = AbletonTrack(new_track_element, self.version_tuple)
        self.tracks.insert(new_position, new_track)
        return new_track

    def _get_max_track_id(self) -> int:
        max_id = 0
        all_tracks = self.root.findall("LiveSet/Tracks/*")
        for track in all_tracks:
            max_id = max(max_id, int(track.get("Id")))
        return max_id
/**
 * 1832 - Check if the Sentence is Pangram
 * Sets | Check existence
 * Time: O(n) | Space: O(1)
 * Space is O(1) because there can only be at most 26 unique characters, which is O(26).
 */

function checkIfPangram(sentence: string): boolean {
  const chars = new Set(sentence);

  return chars.size === 26;
}
